open Base
open Narya
module N = Ndarray
module T = Tensor
module O = Optim.Adam

let bce ?(eps = 1e-6) y y' =
  let open T.Infix in
  let one = T.scalar 1. in
  let eps = T.scalar eps in
  T.relu y' - (y * y') + T.log (one + T.exp (-T.abs y') + eps) |> T.mean
;;

module type Layer = sig
  type t

  val params : ?trainable:bool -> t -> T.t list
end

module Linear : sig
  type t

  include Layer with type t := t

  val create : input_dim:int -> output_dim:int -> t
  val forward : t -> T.t -> T.t
end = struct
  type t =
    { w : T.t
    ; b : T.t
    }

  let he_init ~input_dim ~output_dim =
    let open N.Infix in
    let scale = Float.sqrt (2. /. Float.of_int input_dim) in
    N.randn [| output_dim; input_dim |] * N.s scale
  ;;

  let create ~input_dim ~output_dim =
    let w = he_init ~input_dim ~output_dim |> T.of_ndarray ~requires_grad:true in
    let b = N.zeros [| output_dim |] |> T.of_ndarray ~requires_grad:true in
    { w; b }
  ;;

  let forward { w; b } x =
    let open T.Infix in
    (x @ T.transpose w) + b
  ;;

  let params ?(trainable = true) m =
    let all_p = [ m.w; m.b ] in
    if trainable then List.filter all_p ~f:T.requires_grad else all_p
  ;;
end

module MLP : sig
  type t

  include Layer with type t := t

  val create : unit -> t
  val forward : t -> T.t -> T.t
end = struct
  type t =
    { l1 : Linear.t
    ; l2 : Linear.t
    ; l3 : Linear.t
    }

  let create () =
    let l1 = Linear.create ~input_dim:2 ~output_dim:32 in
    let l2 = Linear.create ~input_dim:32 ~output_dim:32 in
    let l3 = Linear.create ~input_dim:32 ~output_dim:1 in
    { l1; l2; l3 }
  ;;

  let forward { l1; l2; l3 } x =
    x
    |> Linear.forward l1
    |> T.relu
    |> Linear.forward l2
    |> T.relu
    |> Linear.forward l3
  ;;

  let params ?(trainable = true) { l1; l2; l3 } =
    Linear.params ~trainable l1
    @ Linear.params ~trainable l2
    @ Linear.params ~trainable l3
  ;;
end

let seed = 42
let n = 1000
let num_steps = 20_000
let log_every = 500
let lr = 1e-3
let beta1 = 0.9
let beta2 = 0.999
let eps = 1e-8
let noise = 0.08

let log_step step loss accur =
  Stdio.printf
    "step=%05d bce=%8.5f accuracy=%6.2f%%\n%!"
    step
    (T.item loss)
    (100. *. T.item accur)
;;

let () =
  N.seed seed;
  let x, y = Datasets.make_moons ~n ~noise in
  let x = T.of_ndarray x in
  let y = T.of_ndarray y in
  let mlp = MLP.create () in
  let optimizer = O.create ~lr ~beta1 ~beta2 ~eps (MLP.params mlp) in
  for i = 0 to num_steps do
    let open T.Infix in
    O.zero_grad optimizer;
    let logits = MLP.forward mlp x in
    let loss = bce y logits in
    let pred = T.sigmoid logits > T.scalar 0.5 in
    let accur = T.mean (T.scalar 1.0 - T.abs (pred - y)) in
    T.backward loss;
    O.step optimizer;
    if i % log_every = 0 then log_step i loss accur
  done
;;
