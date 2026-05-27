open! Core
module A = Narya.Ndarray

let print_array f v = print_s (Array.sexp_of_t f v)

let%expect_test "create zeros" =
  let x = A.zeros [| 2; 2 |] in
  print_array Float.sexp_of_t (A.to_array x);
  [%expect {| (0 0 0 0) |}]
;;

let%expect_test "scalar shape and sum" =
  let x = A.of_array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let s = A.sum x in
  print_s [%sexp (A.shape s : int array), (A.to_array s : float array)];
  [%expect {| (() (21)) |}]
;;

let%expect_test "reshape preserves logical row-major order" =
  let x = A.of_array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let y = A.reshape x ~shape:[| 3; 2 |] in
  print_array Float.sexp_of_t (A.to_array y);
  [%expect {| (1 2 3 4 5 6) |}]
;;

let%expect_test "transpose logical order" =
  let x = A.of_array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let y = A.transpose x in
  print_s [%sexp (A.shape y : int array), (A.to_array y : float array)];
  [%expect {| ((3 2) (1 4 2 5 3 6)) |}]
;;

let%expect_test "transpose is a mutable view" =
  let x = A.of_array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let y = A.transpose x in
  A.set y [| 1; 0 |] 40.;
  print_array Float.sexp_of_t (A.to_array x);
  [%expect {| (1 40 3 4 5 6) |}]
;;

let%expect_test "map over transposed view uses logical order" =
  let x = A.of_array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let y = A.transpose x |> A.map ~f:(fun v -> v *. 10.) in
  print_s [%sexp (A.shape y : int array), (A.to_array y : float array)];
  [%expect {| ((3 2) (10 40 20 50 30 60)) |}]
;;

let%expect_test "matmul basic" =
  let a = A.of_array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = A.of_array ~shape:[| 3; 2 |] [| 7.; 8.; 9.; 10.; 11.; 12. |] in
  let c = A.matmul a b in
  print_s [%sexp (A.shape c : int array), (A.to_array c : float array)];
  [%expect {| ((2 2) (58 64 139 154)) |}]
;;

let%expect_test "broadcast_shape" =
  print_s [%sexp (A.broadcast_shape [| 3; 1 |] [| 1; 4 |] : int array)];
  print_s [%sexp (A.broadcast_shape [| 3 |] [| 2; 3 |] : int array)];
  print_s [%sexp (A.broadcast_shape [||] [| 2; 3 |] : int array)];
  [%expect
    {|
    (3 4)
    (2 3)
    (2 3) |}]
;;

let%expect_test "broadcast_to repeats with zero strides" =
  let x = A.of_array ~shape:[| 3 |] [| 1.; 2.; 3. |] in
  let y = A.broadcast_to x ~shape:[| 2; 3 |] in
  print_s [%sexp (A.shape y : int array), (A.to_array y : float array)];
  [%expect {| ((2 3) (1 2 3 1 2 3)) |}]
;;

let%expect_test "broadcasted set aliases repeated values" =
  let x = A.of_array ~shape:[| 1; 3 |] [| 1.; 2.; 3. |] in
  let y = A.broadcast_to x ~shape:[| 2; 3 |] in
  A.set y [| 1; 1 |] 20.;
  print_array Float.sexp_of_t (A.to_array x);
  print_array Float.sexp_of_t (A.to_array y);
  [%expect
    {|
    (1 20 3)
    (1 20 3 1 20 3) |}]
;;

let%expect_test "map2 broadcasts vector over matrix" =
  let x =
    A.of_array ~shape:[| 2; 3 |] [| 10.; 20.; 30.; 40.; 50.; 60. |]
  in
  let y = A.of_array ~shape:[| 3 |] [| 1.; 2.; 3. |] in
  let z = A.add x y in
  print_s [%sexp (A.shape z : int array), (A.to_array z : float array)];
  [%expect {| ((2 3) (11 22 33 41 52 63)) |}]
;;

let%expect_test "map2 broadcasts scalar" =
  let x = A.of_array ~shape:[| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let s = A.of_array ~shape:[||] [| 10. |] in
  let z = A.mul x s in
  print_s [%sexp (A.shape z : int array), (A.to_array z : float array)];
  [%expect {| ((2 2) (10 20 30 40)) |}]
;;

let%expect_test "arange" =
  let x = A.arange 5 in
  print_s [%sexp (A.shape x : int array), (A.to_array x : float array)];
  [%expect {| ((5) (0 1 2 3 4)) |}]
;;

let%expect_test "eye" =
  let x = A.eye 3 in
  print_s [%sexp (A.shape x : int array), (A.to_array x : float array)];
  [%expect {| ((3 3) (1 0 0 0 1 0 0 0 1)) |}]
;;

let%expect_test "linspace includes endpoints" =
  let x = A.linspace ~start:0.0 ~stop:1.0 ~num:5 in
  print_s [%sexp (A.shape x : int array), (A.to_array x : float array)];
  [%expect {| ((5) (0 0.25 0.5 0.75 1)) |}]
;;

let%expect_test "linspace with one element returns start" =
  let x = A.linspace ~start:2.0 ~stop:10.0 ~num:1 in
  print_s [%sexp (A.shape x : int array), (A.to_array x : float array)];
  [%expect {| ((1) (2)) |}]
;;

let%expect_test "sum_axis axis 0" =
  let x = A.of_array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let y = A.sum_axis x ~axis:0 in
  print_s [%sexp (A.shape y : int array), (A.to_array y : float array)];
  [%expect {| ((3) (5 7 9)) |}]
;;

let%expect_test "sum_axis axis 1" =
  let x = A.of_array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let y = A.sum_axis x ~axis:1 in
  print_s [%sexp (A.shape y : int array), (A.to_array y : float array)];
  [%expect {| ((2) (6 15)) |}]
;;

let%expect_test "sum_axis keepdim" =
  let x = A.of_array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let y = A.sum_axis x ~axis:1 ~keepdim:true in
  print_s [%sexp (A.shape y : int array), (A.to_array y : float array)];
  [%expect {| ((2 1) (6 15)) |}]
;;

let%expect_test "mean_axis keepdim" =
  let x = A.of_array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let y = A.mean_axis x ~axis:1 ~keepdim:true in
  print_s [%sexp (A.shape y : int array), (A.to_array y : float array)];
  [%expect {| ((2 1) (2 5)) |}]
;;

let%expect_test "sum_to_shape vector broadcast grad" =
  let g = A.ones [| 2; 3 |] in
  let y = A.sum_to_shape g ~shape:[| 3 |] in
  print_s [%sexp (A.shape y : int array), (A.to_array y : float array)];
  [%expect {| ((3) (2 2 2)) |}]
;;

let%expect_test "sum_to_shape singleton dims" =
  let g = A.ones [| 2; 3; 4 |] in
  let y = A.sum_to_shape g ~shape:[| 1; 3; 1 |] in
  print_s [%sexp (A.shape y : int array), (A.to_array y : float array)];
  [%expect {| ((1 3 1) (8 8 8)) |}]
;;

let%expect_test "sum_to_shape scalar" =
  let g = A.ones [| 2; 3 |] in
  let y = A.sum_to_shape g ~shape:[||] in
  print_s [%sexp (A.shape y : int array), (A.to_array y : float array)];
  [%expect {| (() (6)) |}]
;;
