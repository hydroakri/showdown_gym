{
  description = "A Nix-flake-based Python dev environment for showdown_gym (RL assignment)";

  inputs = {
    # Pinned to the same revision already used by showdown_agent/flake.nix
    # in this project (flakehub.com's TLS endpoint has been unreliable here).
    nixpkgs.url = "github:NixOS/nixpkgs/e2587caef70cea85dd97d7daab492899902dbf5d";
  };

  outputs =
    { self, ... }@inputs:

    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forEachSupportedSystem =
        f:
        inputs.nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            inherit system;
            pkgs = import inputs.nixpkgs { inherit system; };
          }
        );
    in
    {
      devShells = forEachSupportedSystem (
        { pkgs, system }:
        {
          default = pkgs.mkShellNoCC {
            packages = with pkgs; [
              python312
              python312Packages.pip
              python312Packages.virtualenv
              self.formatter.${system}
            ];

            # cares_reinforcement_learning[gym] pulls in torch/opencv-python/
            # mujoco/dm_control -- manylinux wheels that dlopen shared libs
            # (libz, libstdc++, GL/X11 for mujoco's renderer) NixOS doesn't
            # expose on the system linker path the way an FHS distro would.
            # Without this, `import numpy` itself fails with "libz.so.1:
            # cannot open shared object file".
            libPath = pkgs.lib.makeLibraryPath (
              with pkgs;
              [
                zlib
                stdenv.cc.cc.lib
                glib
                libGL
                libGLU
                mesa
                xorg.libX11
                xorg.libXext
                xorg.libXrandr
                xorg.libXinerama
                xorg.libXcursor
                xorg.libXi
              ]
            );

            # showdown_gym depends on the sibling cares_reinforcement_learning
            # checkout (../cares_reinforcement_learning) providing the
            # "showdown" gym registration, which in turn imports
            # showdown_gym.showdown_environment -- so both packages install
            # editable into the same venv, cares_rl first per the assignment
            # README's documented install order.
            shellHook = ''
              # /run/opengl-driver/lib carries the system's NVIDIA driver
              # (libcuda.so.1 etc, set up by the NixOS module) -- torch's
              # pip wheel bundles its own CUDA runtime but still needs this
              # driver-side lib to talk to the GPU, or it silently falls
              # back to "Device: cpu".
              export LD_LIBRARY_PATH="$libPath:/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}"

              if [ ! -d .venv ]; then
                python3 -m venv .venv
              fi
              source .venv/bin/activate
              pip install -q --upgrade pip

              if [ -d ../cares_reinforcement_learning ]; then
                pip install -q -e '../cares_reinforcement_learning[gym]'
              else
                echo "warning: ../cares_reinforcement_learning not found -- 'cares-rl' CLI will be unavailable" >&2
              fi

              pip install -q -r requirements.txt
              pip install -q -e .
            '';
          };
        }
      );

      formatter = forEachSupportedSystem ({ pkgs, ... }: pkgs.nixfmt);
    };
}
