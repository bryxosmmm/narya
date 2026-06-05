open! Base

let ndarray = Ndarray.print
let tensor x = x |> Tensor.value |> Ndarray.print

let tensor_grad x =
  match Tensor.grad x with
  | None -> Stdio.print_endline "None"
  | Some grad -> Ndarray.print grad
;;
