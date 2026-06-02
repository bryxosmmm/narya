open! Base
open Basic

module type S = sig
  type t

  val zero_grad : t -> unit
  val step : t -> unit
end

module SGD : sig
  type t

  include S with type t := t

  val create : ?lr:float -> Tensor.t list -> t
end = struct
  type t =
    { params : Tensor.t list
    ; lr : Ndarray.t
    }

  let create ?(lr = 1e-3) params = { params; lr = Ndarray.s lr }
  let zero_grad { params; _ } = List.iter params ~f:Tensor.zero_grad

  let step { params; lr } =
    List.iter params ~f:(fun p ->
      match Tensor.grad p with
      | None -> ()
      | Some g -> Tensor.update p ~f:Ndarray.Infix.(fun v -> v - (lr * g)))
  ;;
end

module SGDM : sig
  type t

  include S with type t := t

  val create : lr:float -> momentum:float -> Tensor.t list -> t
end = struct
  type s =
    { p : Tensor.t
    ; mutable v : Ndarray.t
    }

  type t =
    { state : s list
    ; alpha : Ndarray.t
    ; beta : Ndarray.t
    }

  let create ~lr ~momentum params =
    let state =
      List.map params ~f:(fun p ->
        let v = p |> Tensor.value |> Ndarray.zeros_like in
        { p; v })
    in
    let alpha = Ndarray.s lr in
    let beta = Ndarray.s momentum in
    { state; alpha; beta }
  ;;

  let zero_grad { state; _ } =
    List.iter state ~f:(fun x -> Tensor.zero_grad x.p)
  ;;

  let step { state; alpha; beta } =
    let open Ndarray.Infix in
    List.iter state ~f:(fun st ->
      match Tensor.grad st.p with
      | None -> ()
      | Some g ->
        st.v <- (beta * st.v) + (alpha * g);
        Tensor.update st.p ~f:Ndarray.Infix.(fun w -> w - st.v))
  ;;
end

module RMSProp : sig
  type t

  include S with type t := t

  val create : lr:float -> decay:float -> eps:float -> Tensor.t list -> t
end = struct
  type s =
    { p : Tensor.t
    ; mutable v : Ndarray.t
    }

  type t =
    { state : s list
    ; alpha : Ndarray.t
    ; rho : Ndarray.t
    ; eps : Ndarray.t
    }

  let create ~lr ~decay ~eps params =
    let state =
      List.map params ~f:(fun p ->
        let v = p |> Tensor.value |> Ndarray.zeros_like in
        { p; v })
    in
    let alpha = Ndarray.s lr in
    let rho = Ndarray.s decay in
    let eps = Ndarray.s eps in
    { state; alpha; rho; eps }
  ;;

  let zero_grad { state; _ } =
    List.iter state ~f:(fun x -> Tensor.zero_grad x.p)
  ;;

  let step { state; alpha; rho; eps } =
    let open Ndarray.Infix in
    List.iter state ~f:(fun st ->
      match Tensor.grad st.p with
      | None -> ()
      | Some g ->
        st.v <- (rho * st.v) + ((Ndarray.s 1. - rho) * (g ^ 2.));
        Tensor.update
          st.p
          ~f:
            Ndarray.Infix.(
              let u = (st.v + eps) ^ -0.5 in
              fun w -> w - (alpha * g * u)))
  ;;
end
