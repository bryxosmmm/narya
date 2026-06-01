open Base
open Narya

let y = Ndarray.arange 10 |> Tensor.of_ndarray

let y' =
  Ndarray.linspace ~start:0.0 ~stop:1.0 ~num:10
  |> Tensor.of_ndarray ~requires_grad:true
;;

let rmse =
  let open Tensor in
  let open Infix in
  (y - y') ^ 2. |> mean |> sqrt
;;

let () =
  Stdio.print_endline "Before backward: ";
  Stdio.print_endline (Tensor.Debug.to_dot rmse);
  Tensor.backward rmse;
  Stdio.print_endline "After backward: ";
  Stdio.print_endline (Tensor.Debug.to_dot rmse)
;;
