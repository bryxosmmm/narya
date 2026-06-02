open Base
open Narya
open Optim
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

let b = N.s 0. |> T.of_ndarray ~requires_grad:true
let opt = RMSProp.create ~lr:1e-4 ~decay:0.8 ~eps:1e-9 [ b; w ]

let () =
  let open T.Infix in
  for step = 0 to 100000 do
    RMSProp.zero_grad opt;
    let y' = (x @ w) + b in
    let loss = (y - y') ^ 2. |> T.mean in
    T.backward loss;
    RMSProp.step opt;
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
