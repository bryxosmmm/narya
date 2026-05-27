open! Core
module T = Narya.Tensor

let%expect_test "tensor tests placeholder" =
  ignore (T.scalar 0.0 : T.t);
  [%expect {||}]
;;
