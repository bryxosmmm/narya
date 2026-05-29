open! Core
module A = Narya.Ndarray
module T = Narya.Tensor

let print_ndarray x = print_s [%sexp (A.shape x : int array), (A.to_array x : float array)]
let print_tensor_value x = print_ndarray (T.value x)
let print_tensor_grad x = print_ndarray (T.grad_exn x)

let%expect_test "scalar add value" =
  let x = T.scalar 2.0 in
  let y = T.scalar 3.0 in
  print_tensor_value (T.add x y);
  [%expect {| (() (5)) |}]
;;

let%expect_test "scalar neg value" =
  let x = T.scalar 2.0 in
  print_tensor_value (T.neg x);
  [%expect {| (() (-2)) |}]
;;

let%expect_test "scalar sub value" =
  let x = T.scalar 2.0 in
  let y = T.scalar 3.0 in
  print_tensor_value (T.sub x y);
  [%expect {| (() (-1)) |}]
;;

let%expect_test "scalar mul value" =
  let x = T.scalar 2.0 in
  let y = T.scalar 3.0 in
  print_tensor_value (T.mul x y);
  [%expect {| (() (6)) |}]
;;

let%expect_test "sum and mean values" =
  let x = T.create ~shape:[| 2; 2 |] 2.0 in
  print_tensor_value (T.sum x);
  print_tensor_value (T.mean x);
  [%expect {|
    (() (8))
    (() (2)) |}]
;;

let%expect_test "backward add" =
  let x = T.scalar ~requires_grad:true 2.0 in
  let y = T.scalar ~requires_grad:true 3.0 in
  let z = T.add x y in
  T.backward z;
  print_tensor_grad x;
  print_tensor_grad y;
  [%expect {|
    (() (1))
    (() (1)) |}]
;;

let%expect_test "backward neg" =
  let x = T.scalar ~requires_grad:true 2.0 in
  let y = T.neg x in
  T.backward y;
  print_tensor_grad x;
  [%expect {| (() (-1)) |}]
;;

let%expect_test "backward sub" =
  let x = T.scalar ~requires_grad:true 2.0 in
  let y = T.scalar ~requires_grad:true 3.0 in
  let z = T.sub x y in
  T.backward z;
  print_tensor_grad x;
  print_tensor_grad y;
  [%expect {|
    (() (1))
    (() (-1)) |}]
;;

let%expect_test "backward mul" =
  let x = T.scalar ~requires_grad:true 2.0 in
  let y = T.scalar ~requires_grad:true 3.0 in
  let z = T.mul x y in
  T.backward z;
  print_tensor_grad x;
  print_tensor_grad y;
  [%expect {|
    (() (3))
    (() (2)) |}]
;;

let%expect_test "backward sum" =
  let x = T.create ~requires_grad:true ~shape:[| 2; 2 |] 2.0 in
  let y = T.sum x in
  T.backward y;
  print_tensor_grad x;
  [%expect {| ((2 2) (1 1 1 1)) |}]
;;

let%expect_test "backward mean" =
  let x = T.create ~requires_grad:true ~shape:[| 2; 2 |] 2.0 in
  let y = T.mean x in
  T.backward y;
  print_tensor_grad x;
  [%expect {| ((2 2) (0.25 0.25 0.25 0.25)) |}]
;;

let%expect_test "backward broadcast add" =
  let x = T.create ~requires_grad:true ~shape:[| 2; 3 |] 1.0 in
  let b = T.create ~requires_grad:true ~shape:[| 3 |] 2.0 in
  let y = T.sum (T.add x b) in
  T.backward y;
  print_tensor_grad x;
  print_tensor_grad b;
  [%expect {|
    ((2 3) (1 1 1 1 1 1))
    ((3) (2 2 2)) |}]
;;

let%expect_test "backward shared parent accumulates once per path" =
  let x = T.scalar ~requires_grad:true 2.0 in
  let y = T.mul x x in
  let z = T.add y y in
  T.backward z;
  print_tensor_grad x;
  [%expect {| (() (8)) |}]
;;

let%expect_test "chain rule square plus identity" =
  let x = T.scalar ~requires_grad:true 3.0 in
  let y = T.add (T.mul x x) x in
  T.backward y;
  print_tensor_value y;
  print_tensor_grad x;
  [%expect {|
    (() (12))
    (() (7)) |}]
;;

let%expect_test "mean of square gradient" =
  let x = T.of_ndarray ~requires_grad:true (A.of_array ~shape:[| 2 |] [| 2.; 4. |]) in
  let y = T.mean (T.mul x x) in
  T.backward y;
  print_tensor_value y;
  print_tensor_grad x;
  [%expect {|
    (() (10))
    ((2) (2 4)) |}]
;;

let%expect_test "constant parent does not receive grad" =
  let x = T.scalar ~requires_grad:true 2.0 in
  let c = T.scalar 3.0 in
  let y = T.mul x c in
  T.backward y;
  print_tensor_grad x;
  print_s [%sexp (Option.is_none (T.grad c) : bool)];
  [%expect {|
    (() (3))
    true
    |}]
;;

let%expect_test "branching graph accumulates gradients" =
  let x = T.scalar ~requires_grad:true 4.0 in
  let two = T.scalar 2.0 in
  let three = T.scalar 3.0 in
  let y = T.add (T.mul x two) (T.mul x three) in
  T.backward y;
  print_tensor_value y;
  print_tensor_grad x;
  [%expect {|
    (() (20))
    (() (5)) |}]
;;

let%expect_test "broadcast multiply gradient for vector factor" =
  let x =
    T.of_ndarray
      ~requires_grad:true
      (A.of_array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |])
  in
  let b = T.of_ndarray ~requires_grad:true (A.of_array ~shape:[| 3 |] [| 10.; 20.; 30. |]) in
  let y = T.sum (T.mul x b) in
  T.backward y;
  print_tensor_value y;
  print_tensor_grad x;
  print_tensor_grad b;
  [%expect {|
    (() (460))
    ((2 3) (10 20 30 10 20 30))
    ((3) (5 7 9))
    |}]
;;

let%expect_test "non-scalar backward seeds ones_like" =
  let x =
    T.of_ndarray
      ~requires_grad:true
      (A.of_array ~shape:[| 2; 2 |] [| 1.; 2.; 3.; 4. |])
  in
  let y = T.mul x x in
  T.backward y;
  print_tensor_grad x;
  [%expect {| ((2 2) (2 4 6 8)) |}]
;;

let%expect_test "relu forward clamps at zero" =
  let x = T.of_ndarray (A.of_array ~shape:[| 5 |] [| -2.; -0.5; 0.; 0.5; 3. |]) in
  let y = T.relu x in
  print_tensor_value y;
  [%expect {| ((5) (0 0 0 0.5 3)) |}]
;;

let%expect_test "relu backward uses positive mask" =
  let x =
    T.of_ndarray
      ~requires_grad:true
      (A.of_array ~shape:[| 5 |] [| -2.; -0.5; 0.; 0.5; 3. |])
  in
  let y = T.sum (T.relu x) in
  T.backward y;
  print_tensor_grad x;
  [%expect {| ((5) (0 0 0 1 1)) |}]
;;

let%expect_test "matmul forward" =
  let a = T.of_ndarray (A.of_array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]) in
  let b = T.of_ndarray (A.of_array ~shape:[| 3; 2 |] [| 7.; 8.; 9.; 10.; 11.; 12. |]) in
  let c = T.matmul a b in
  print_tensor_value c;
  [%expect {| ((2 2) (58 64 139 154)) |}]
;;

let%expect_test "matmul backward" =
  let a =
    T.of_ndarray
      ~requires_grad:true
      (A.of_array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |])
  in
  let b =
    T.of_ndarray
      ~requires_grad:true
      (A.of_array ~shape:[| 3; 2 |] [| 7.; 8.; 9.; 10.; 11.; 12. |])
  in
  let loss = T.sum (T.matmul a b) in
  T.backward loss;
  print_tensor_grad a;
  print_tensor_grad b;
  [%expect {|
    ((2 3) (15 19 23 15 19 23))
    ((3 2) (5 5 7 7 9 9)) |}]
;;

let%expect_test "div forward and backward" =
  let x = T.scalar ~requires_grad:true 6.0 in
  let y = T.scalar ~requires_grad:true 3.0 in
  let z = T.div x y in
  T.backward z;
  print_tensor_value z;
  print_tensor_grad x;
  print_tensor_grad y;
  [%expect {|
    (() (2))
    (() (0.33333333333333331))
    (() (-0.66666666666666663))
    |}]
;;

let%expect_test "square forward and backward" =
  let x = T.scalar ~requires_grad:true 3.0 in
  let y = T.square x in
  T.backward y;
  print_tensor_value y;
  print_tensor_grad x;
  [%expect {|
    (() (9))
    (() (6)) |}]
;;

let%expect_test "powf forward and backward" =
  let x = T.scalar ~requires_grad:true 2.0 in
  let y = T.powf x 3.0 in
  T.backward y;
  print_tensor_value y;
  print_tensor_grad x;
  [%expect {|
    (() (8))
    (() (12)) |}]
;;

let%expect_test "sum_axis backward" =
  let x =
    T.of_ndarray
      ~requires_grad:true
      (A.of_array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |])
  in
  let y = T.sum (T.sum_axis x ~axis:1) in
  T.backward y;
  print_tensor_grad x;
  [%expect {| ((2 3) (1 1 1 1 1 1)) |}]
;;

let%expect_test "sum_axis keepdim backward" =
  let x =
    T.of_ndarray
      ~requires_grad:true
      (A.of_array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |])
  in
  let y = T.sum (T.sum_axis x ~axis:1 ~keepdim:true) in
  T.backward y;
  print_tensor_grad x;
  [%expect {| ((2 3) (1 1 1 1 1 1)) |}]
;;

let%expect_test "mean_axis backward" =
  let x =
    T.of_ndarray
      ~requires_grad:true
      (A.of_array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |])
  in
  let y = T.sum (T.mean_axis x ~axis:1) in
  T.backward y;
  print_tensor_grad x;
  [%expect {| ((2 3) (0.333333333333333315 0.333333333333333315 0.333333333333333315 0.333333333333333315 0.333333333333333315 0.333333333333333315)) |}]
;;

let%expect_test "mean_axis keepdim backward" =
  let x =
    T.of_ndarray
      ~requires_grad:true
      (A.of_array ~shape:[| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |])
  in
  let y = T.sum (T.mean_axis x ~axis:1 ~keepdim:true) in
  T.backward y;
  print_tensor_grad x;
  [%expect {| ((2 3) (0.333333333333333315 0.333333333333333315 0.333333333333333315 0.333333333333333315 0.333333333333333315 0.333333333333333315)) |}]
;;
