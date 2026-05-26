open! Base

type t 

val create: shape:int array -> requires_grad:bool -> float -> t
val zeros: shape:int array -> requires_grad:bool -> t
val ones: shape:int array -> requires_grad:bool -> t
val scalar: requires_grad:bool -> float -> t

val value: t -> Ndarray.t
val grad: t -> Ndarray.t option
val grad_exn: t -> Ndarray.t
val requires_grad: t -> bool






