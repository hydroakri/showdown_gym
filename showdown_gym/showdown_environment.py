import os
import time
from typing import Any, Dict
import numpy as np

from poke_env import (
    AccountConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import ObsType
from poke_env.player.player import Player
from .base_environment import BaseShowdownEnv

from poke_env.data import GenData

# Load Gen9 data (type chart etc.)
GEN_DATA = GenData.from_gen(9)


class ShowdownEnvironment(BaseShowdownEnv):

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )
        self.rl_agent = account_name_one
        self._prev_battle_state = {}

    def reset(self, seed=None, options=None):
        # poke_env's Player._battles dict keeps every finished battle
        # forever -- poke_env only clears it via env.close(purge=True),
        # which nothing in the training loop ever calls. Over a long
        # training run (thousands of episodes) this accumulates several
        # GB of retained AbstractBattle history that nothing here needs:
        # calc_reward/embed_battle receive the live battle object
        # directly, and _get_prior_battle compares against separately
        # deepcopy'd snapshots, not this dict. Safe to purge every reset,
        # since only already-finished battles are in it at this point
        # (the new episode's battle doesn't exist until super().reset()
        # below creates it).
        try:
            self.agent1.reset_battles()
            self.agent2.reset_battles()
        except EnvironmentError:
            pass  # a battle wasn't marked finished yet -- skip this cycle

        return super().reset(seed, options)

    # =========================================================
    # Action space
    # =========================================================
    # poke_env's default 26-action mapping reserves action indices 10-21
    # for "move + mega-evolve / z-move / dynamax" combos (num_gimmicks=4 in
    # SinglesEnv.__init__ for any gen9 format). None of those three
    # mechanics exist in Gen9 Ubers (Dynamax was removed after Gen8,
    # Mega Evolution and Z-Moves after Gen7) -- battle.can_mega_evolve,
    # battle.can_z_move, and battle.can_dynamax are permanently False for
    # every battle this agent ever plays. Those 12 indices (46% of the
    # action space) are therefore *always* illegal: action_to_order's
    # strict=False fallback silently rewrites them to "/choose default"
    # (SinglesEnv is constructed with strict=False in BaseShowdownEnv),
    # which executes a different move than the one the network selected.
    # That corrupts the (state, action, reward) tuple stored in the replay
    # buffer -- the TD update's q_value(s, a) target is built from a reward
    # that didn't actually come from a. Trimming the network's action
    # space to the 14 indices that can ever be legal (6 switches + 4 moves
    # + 4 move+terastallize, skipping the dead mega/z-move/dynamax block)
    # removes this corruption at the source instead of hoping the network
    # learns the dead zone purely from reward signal.
    def _get_action_size(self) -> int | None:
        return 14

    def process_action(self, action: np.int64) -> np.int64:
        # 0-5: switch, 6-9: move (unchanged). 10-13 (our trimmed space's
        # "move + terastallize") map to the real action space's 22-25.
        if action >= 10:
            return np.int64(action + 12)
        return action

    # =========================================================
    # Reward Function
    # =========================================================
    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Reward based on HP, fainted Pokémon, and victory outcomes.
        Inspired by SimpleRLPlayer reward_computing_helper.
        """
        prior_battle = self._get_prior_battle(battle)

        if battle is None:
            return 0.0

   
        ally_hp = np.sum([m.current_hp_fraction for m in battle.team.values()])
        opp_hp = np.sum([m.current_hp_fraction for m in battle.opponent_team.values()])
        hp_diff = ally_hp - opp_hp

        ally_fainted = sum(m.fainted for m in battle.team.values())
        opp_fainted = sum(m.fainted for m in battle.opponent_team.values())
        faint_diff = (opp_fainted - ally_fainted) * 2.0

        if prior_battle:
            prev_ally_hp = np.sum([m.current_hp_fraction for m in prior_battle.team.values()])
            prev_opp_hp = np.sum([m.current_hp_fraction for m in prior_battle.opponent_team.values()])
            hp_delta = (prev_opp_hp - opp_hp) - (prev_ally_hp - ally_hp)
        else:
            hp_delta = 0.0

        victory_bonus = 0.0
        if battle.finished:
            if battle.won:
                victory_bonus += 30.0
            elif battle.lost:
                victory_bonus -= 15.0

        reward = 1.0 * hp_delta + 0.5 * hp_diff + faint_diff + victory_bonus
        return float(np.clip(reward, -30.0, 30.0))

    # =========================================================
    # Observation space
    # =========================================================
    def _observation_size(self) -> int:
        """
        Embedding structure:
          4x base powers
          4x type multipliers
          2x (ally_fainted/6, opp_fainted/6)
          2x (ally_total_hp, opp_total_hp)
          1x ally_faster (base speed comparison)
          2x (ally_active_has_status, opp_active_has_status)
          3x (best_switch_offense, best_switch_defense, bench_alive_frac)
        = 18 features

        The last 6 features exist because the action space includes
        switching (indices 0-5), but nothing in the original 12-feature
        state described the *bench* -- the network could select a switch
        action but had no information to judge which of its other 5 team
        members would be a good matchup, or whether switching is even
        attractive right now (ally_faster/status). Added as bounded
        aggregates (max/min pooling over bench Pokemon) rather than
        per-slot features, since team order isn't semantically meaningful
        across battles and this keeps the feature count fixed regardless
        of how many bench members are still alive.
        """
        return 18

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        SB3-style compact embedding of the battle state.
        Combines per-move offensive info with overall HP context and
        switch-relevant bench information.
        """
        obs = np.zeros(self._observation_size(), dtype=np.float32)

        if not battle.active_pokemon or not battle.opponent_active_pokemon:
            return obs

        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        moves_base_power = np.zeros(4, dtype=np.float32)
        moves_dmg_multiplier = np.ones(4, dtype=np.float32)

        for i, move in enumerate(battle.available_moves[:4]):
            moves_base_power[i] = float((move.base_power or 0) / 100.0)
            try:
                if move.type and opp.type_1:
                    mult = move.type.damage_multiplier(
                        opp.type_1, getattr(opp, "type_2", None),
                        type_chart=GEN_DATA.type_chart,
                    )
                    moves_dmg_multiplier[i] = float(mult)
            except Exception:
                moves_dmg_multiplier[i] = 1.0

        ally_hp_total = np.sum([m.current_hp_fraction for m in battle.team.values()]) / 6.0
        opp_hp_total = np.sum([m.current_hp_fraction for m in battle.opponent_team.values()]) / 6.0
        ally_fainted = len([m for m in battle.team.values() if m.fainted]) / 6.0
        opp_fainted = len([m for m in battle.opponent_team.values() if m.fainted]) / 6.0

        try:
            ally_faster = float(active.base_stats["spe"] > opp.base_stats["spe"])
        except Exception:
            ally_faster = 0.5

        ally_has_status = float(active.status is not None)
        opp_has_status = float(opp.status is not None)

        bench = [m for m in battle.team.values() if m is not active and not m.fainted]
        best_switch_offense = 0.0
        best_switch_defense = 1.0  # 1.0 = neutral; lower found below = safer
        if bench:
            offense_scores = []
            defense_scores = []
            for mon in bench:
                try:
                    off = mon.type_1.damage_multiplier(
                        opp.type_1, getattr(opp, "type_2", None),
                        type_chart=GEN_DATA.type_chart,
                    )
                    if mon.type_2 is not None:
                        off = max(off, mon.type_2.damage_multiplier(
                            opp.type_1, getattr(opp, "type_2", None),
                            type_chart=GEN_DATA.type_chart,
                        ))
                    offense_scores.append(off)
                except Exception:
                    pass
                try:
                    dfn = opp.type_1.damage_multiplier(
                        mon.type_1, mon.type_2, type_chart=GEN_DATA.type_chart,
                    )
                    if opp.type_2 is not None:
                        dfn = max(dfn, opp.type_2.damage_multiplier(
                            mon.type_1, mon.type_2, type_chart=GEN_DATA.type_chart,
                        ))
                    defense_scores.append(dfn)
                except Exception:
                    pass
            if offense_scores:
                best_switch_offense = max(offense_scores)
            if defense_scores:
                best_switch_defense = min(defense_scores)

        bench_alive_frac = len(bench) / 5.0

        obs = np.concatenate([
            moves_base_power,
            moves_dmg_multiplier,
            np.array([
                ally_fainted, opp_fainted, ally_hp_total, opp_hp_total,
                ally_faster, ally_has_status, opp_has_status,
                min(best_switch_offense / 4.0, 1.0),
                min(best_switch_defense / 4.0, 1.0),
                bench_alive_frac,
            ], dtype=np.float32),
        ])

        return obs.astype(np.float32)

    # =========================================================
    # Additional info (logging)
    # =========================================================
    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()
        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won
            info[agent]["turns"] = self.battle1.turn
        return info



########################################
# DO NOT EDIT BELOW THIS LINE
########################################

class SingleShowdownWrapper(SingleAgentWrapper):
    """
    Wrapper for single-agent training against specified opponents.
    """

    def __init__(self, team_type: str = "random", opponent_type: str = "random", evaluation: bool = False):
        opponent: Player
        unique_id = time.strftime("%H%M%S")

        opponent_account = "ot" if not evaluation else "oe"
        opponent_account = f"{opponent_account}_{unique_id}"

        opponent_configuration = AccountConfiguration(opponent_account, None)
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "random":
            opponent = RandomPlayer(account_configuration=opponent_configuration)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "t1" if not evaluation else "e1"
        account_name_two: str = "t2" if not evaluation else "e2"
        account_name_one = f"{account_name_one}_{unique_id}"
        account_name_two = f"{account_name_two}_{unique_id}"

        team = self._load_team(team_type)
        battle_format = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        super().__init__(env=primary_env, opponent=opponent)

    def _load_team(self, team_type: str) -> str | None:
        bot_teams_folders = os.path.join(os.path.dirname(__file__), "teams")
        bot_teams = {}
        for team_file in os.listdir(bot_teams_folders):
            if team_file.endswith(".txt"):
                with open(os.path.join(bot_teams_folders, team_file), "r", encoding="utf-8") as file:
                    bot_teams[team_file[:-4]] = file.read()
        return bot_teams.get(team_type, None)
