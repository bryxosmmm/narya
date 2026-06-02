(lang dune 3.21)

(name narya)

(generate_opam_files true)

(formatting
 (enabled_for ocaml))

(source
 (github bryxosmmm/narya))

(authors "bryxosmmm <bryxosmmm@proton.me>")

(maintainers "bryxosmmm <bryxosmmm@proton.me>")

(license MIT)

(documentation https://github.com/bryxosmmm/narya)

(package
 (name narya)
 (synopsis "Experimental tensor and autodiff library in OCaml")
 (description
  "Narya is an experimental machine learning library written in OCaml. It implements numerical arrays, tensors, reverse-mode automatic differentiation, and basic optimizers from first principles.")
 (depends
  (ocaml (>= 5.0))
  dune
  base
  core
  stdio
  ppx_jane
  ppx_expect)
 (tags
  (ocaml machine-learning tensors autodiff optimizers)))
