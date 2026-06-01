open Base
open Narya
module T = Tensor
module N = Ndarray

let () = N.seed 42
let f x = (2. *. x) +. 1.
let x = N.arange 100 |> N.unsqueeze ~axis:1 |> T.of_ndarray
let y = N.arange 100 |> N.map ~f |> N.unsqueeze ~axis:1 |> T.of_ndarray

let w =
  N.Infix.(N.randn [| 1; 1 |] * N.s 0.01)
  |> T.of_ndarray ~requires_grad:true
;;

let u = N.zeros [| 1; 1 |] |> T.of_ndarray
let b = N.s 0. |> T.of_ndarray ~requires_grad:true
let lr = N.s 1e-4
let beta = N.s 0.1

let momentum v g =
  let open N.Infix in
  (T.value v * beta) + (lr * g)
;;

let () =
  let open T.Infix in
  for step = 0 to 100000 do
    T.zero_grad w;
    T.zero_grad b;
    let y' = (x @ w) + b in
    let loss = (y - y') ^ 2. |> T.mean in
    T.backward loss;
    let gw = T.grad w |> Option.value_exn in
    let gb = T.grad b |> Option.value_exn in
    T.update w ~f:N.Infix.(fun v -> v - momentum mv gw);
    T.update b ~f:N.Infix.(fun v -> v - (lr * gb));
    if step % 1000 = 0
    then
      Stdio.printf
        "step=%04d loss=%f w=%f b=%f\n%!"
        step
        (T.item loss)
        (T.item w)
        (T.item b)
  done
;;
