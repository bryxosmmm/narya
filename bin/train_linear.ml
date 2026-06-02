open Base
open Narya
module N = Ndarray
module T = Tensor
module O = Optim.SGDM

let seed = 42
let num_samples = 500
let num_steps = 10_000
let log_every = 1000
let learning_rate = 1e-3
let momentum = 0.9
let a = 2.0
let c = 1.0
let eps = 1e-12
let f x = (a *. x) +. c

let make_dataset n =
  let x =
    N.linspace ~start:(-1.0) ~stop:1.0 ~num:n |> N.unsqueeze ~axis:1
  in
  let y = x |> N.map ~f in
  T.of_ndarray x, T.of_ndarray y
;;

let create_model () =
  let w =
    N.Infix.(N.randn [| 1; 1 |] * N.s 0.01)
    |> T.of_ndarray ~requires_grad:true
  in
  let b = N.s 0. |> T.of_ndarray ~requires_grad:true in
  w, b
;;

let forward ~x ~w ~b =
  let open T.Infix in
  (x @ w) + b
;;

let rmse y prediction =
  let open T.Infix in
  let mse = (y - prediction) ^ 2. |> T.mean in
  T.sqrt (mse + T.scalar eps)
;;

let train_step optimizer ~x ~y ~w ~b =
  O.zero_grad optimizer;
  let y' = forward ~x ~w ~b in
  let loss = rmse y y' in
  T.backward loss;
  O.step optimizer;
  loss
;;

let log_step step loss ~w ~b =
  Stdio.printf
    "step=%05d rmse=%8.5f w=%8.5f b=%8.5f\n%!"
    step
    (T.item loss)
    (T.item w)
    (T.item b)
;;

let () =
  N.seed seed;
  let x, y = make_dataset num_samples in
  let w, b = create_model () in
  let optimizer = O.create ~lr:learning_rate ~momentum [ w; b ] in
  Stdio.printf "Training y = %.1fx + %.1f with SGDM\n%!" a c;
  for step = 0 to num_steps do
    let loss = train_step optimizer ~x ~y ~w ~b in
    if step % log_every = 0 then log_step step loss ~w ~b
  done;
  Stdio.printf "Done: learned y = %.5fx + %.5f\n%!" (T.item w) (T.item b)
;;
