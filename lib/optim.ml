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
        st.v <- (beta * st.v) + (alpha * g);
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
