open! Base

type t

val create : ?requires_grad:bool -> shape:int array -> float -> t
val zeros : ?requires_grad:bool -> int array -> t
val ones : ?requires_grad:bool -> int array -> t
val scalar : ?requires_grad:bool -> float -> t
val of_ndarray : ?requires_grad:bool -> Ndarray.t -> t
val value : t -> Ndarray.t
val grad : t -> Ndarray.t option
val grad_exn : t -> Ndarray.t
val requires_grad : t -> bool

val add : t -> t -> t
val sub : t -> t -> t
val mul : t -> t -> t
val neg : t -> t
val sum : t -> t
val mean : t -> t

val backward : t -> unit
