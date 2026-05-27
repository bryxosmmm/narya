open! Base

type t

val create : ?requires_grad:bool -> shape:int array -> float -> t
val zeros : ?requires_grad:bool -> int array -> t
val ones : ?requires_grad:bool -> int array -> t
val scalar : ?requires_grad:bool -> float -> t
val value : t -> Ndarray.t
val grad : t -> Ndarray.t option
val grad_exn : t -> Ndarray.t
val requires_grad : t -> bool
