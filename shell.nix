{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = with pkgs; [ 

    # OpenCV with GUI support
    (python3Packages.opencv4.override { enableGtk2 = true; })
  
  ];
  shellHook = "source .venv/bin/activate";
}
