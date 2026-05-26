open! Core
module A = Narya.Ndarray
module T = Narya.Tensor

let print_array f v = print_s (Array.sexp_of_t f v)

let%expect_test "create zeros" =
  let x = A.zeros [| 2; 2 |] in
  print_array Float.sexp_of_t (A.to_array x);
  [%expect {| (0 0 0 0) |}]
;;
