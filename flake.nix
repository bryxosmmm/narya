{
  description = "Development shell with uv and Python 3.13";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            ocamlPackages.ocaml
            ocamlPackages.dune_3
            ocamlPackages.findlib
            ocamlPackages.ocaml-lsp
            ocamlPackages.ocamlformat
            ocamlPackages.utop
            ocamlPackages.base
            ocamlPackages.stdio
            ocamlPackages.core
            ocamlPackages.ppx_jane
            ocamlPackages.ppx_expect
            ocamlPackages.alcotest
            ocamlPackages.cmdliner
          ];
        };
      }
    );
}
