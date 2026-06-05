open! Base
open Basic
module N = Ndarray
module T = Tensor

module type S = sig
  type t

  val zero_grad : t -> unit
  val step : t -> unit
end

module SGD : sig
  type t

  include S with type t := t

  val create : ?lr:float -> T.t list -> t
end = struct
  type t =
    { params : T.t list
    ; lr : N.t
    }

  let create ?(lr = 1e-3) params = { params; lr = N.s lr }
  let zero_grad { params; _ } = List.iter params ~f:T.zero_grad

  let step { params; lr } =
    List.iter params ~f:(fun p ->
      match T.grad p with
      | None -> ()
      | Some g -> T.update p ~f:N.Infix.(fun v -> v - (lr * g)))
  ;;
end

module SGDM : sig
  type t

  include S with type t := t

  val create : lr:float -> momentum:float -> T.t list -> t
end = struct
  type s =
    { p : T.t
    ; mutable v : N.t
    }

  type t =
    { state : s list
    ; alpha : N.t
    ; beta : N.t
    }

  let create ~lr ~momentum params =
    let state =
      List.map params ~f:(fun p ->
        let v = p |> T.value |> N.zeros_like in
        { p; v })
    in
    let alpha = N.s lr in
    let beta = N.s momentum in
    { state; alpha; beta }
  ;;

  let zero_grad { state; _ } =
    List.iter state ~f:(fun x -> T.zero_grad x.p)
  ;;

  let step { state; alpha; beta } =
    let open N.Infix in
    List.iter state ~f:(fun st ->
      match T.grad st.p with
      | None -> ()
      | Some g ->
        st.v <- (beta * st.v) + ((N.s 1. - beta) * (alpha * g));
        T.update st.p ~f:N.Infix.(fun w -> w - st.v))
  ;;
end

module RMSProp : sig
  type t

  include S with type t := t

  val create : lr:float -> decay:float -> eps:float -> T.t list -> t
end = struct
  type s =
    { p : T.t
    ; mutable v : N.t
    }

  type t =
    { state : s list
    ; alpha : N.t
    ; rho : N.t
    ; eps : N.t
    }

  let create ~lr ~decay ~eps params =
    let state =
      List.map params ~f:(fun p ->
        let v = p |> T.value |> N.zeros_like in
        { p; v })
    in
    let alpha = N.s lr in
    let rho = N.s decay in
    let eps = N.s eps in
    { state; alpha; rho; eps }
  ;;

  let zero_grad { state; _ } =
    List.iter state ~f:(fun x -> T.zero_grad x.p)
  ;;

  let step { state; alpha; rho; eps } =
    let open N.Infix in
    List.iter state ~f:(fun st ->
      match T.grad st.p with
      | None -> ()
      | Some g ->
        st.v <- (rho * st.v) + ((N.s 1. - rho) * (g ^ 2.));
        T.update
          st.p
          ~f:
            N.Infix.(
              let u = (st.v + eps) ^ -0.5 in
              fun w -> w - (alpha * g * u)))
  ;;
end

module Adam : sig
  type t

  include S with type t := t

  val create
    :  lr:float
    -> beta1:float
    -> beta2:float
    -> eps:float
    -> T.t list
    -> t
end = struct
  type s =
    { p : T.t
    ; mutable v : N.t
    ; mutable m : N.t
    }

  type t =
    { state : s list
    ; alpha : N.t
    ; beta1 : N.t
    ; beta2 : N.t
    ; eps : N.t
    ; mutable step_cnt : int
    }

  let create ~lr ~beta1 ~beta2 ~eps params =
    let state =
      List.map params ~f:(fun p ->
        let v = p |> T.value |> N.zeros_like in
        let m = p |> T.value |> N.zeros_like in
        { p; v; m })
    in
    let alpha = N.s lr in
    let beta1 = N.s beta1 in
    let beta2 = N.s beta2 in
    let eps = N.s eps in
    let step_cnt = 0 in
    { state; alpha; beta1; beta2; eps; step_cnt }
  ;;

  let zero_grad { state; _ } =
    List.iter state ~f:(fun x -> T.zero_grad x.p)
  ;;

  let step o =
    o.step_cnt <- o.step_cnt + 1;
    let k = Float.of_int o.step_cnt in
    let open N.Infix in
    List.iter o.state ~f:(fun st ->
      match T.grad st.p with
      | None -> ()
      | Some g ->
        st.m <- (o.beta1 * st.m) + ((N.s 1. - o.beta1) * g);
        st.v <- (o.beta2 * st.v) + ((N.s 1. - o.beta2) * (g ^ 2.));
        T.update
          st.p
          ~f:
            N.Infix.(
              let m' = st.m / (N.s 1. - (o.beta1 ^ k)) in
              let v' = st.v / (N.s 1. - (o.beta2 ^ k)) in
              let u = (v' ^ 0.5) + o.eps in
              fun w -> w - (o.alpha * m' / u)))
  ;;
end
