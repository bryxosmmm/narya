open! Base

type t

val create : int array -> float -> t
val zeros : int array -> t
val ones : int array -> t

val of_array : shape:int array -> float array -> t
val to_array: t -> float array

val shape : t -> int array
val ndim : t -> int
val numel : t -> int

val get : t -> int array -> float
val set : t -> int array -> float -> unit

val update : t -> int array -> f:(float -> float) -> unit

val map : t -> f:(float -> float) -> t
val map2 : t -> t -> f:(float -> float -> float) -> t

val add : t -> t -> t
val sub : t -> t -> t
val mul : t -> t -> t
val div : t -> t -> t
val neg : t -> t
val add' : t -> float -> t
val sub' : t -> float -> t
val mul' : t -> float -> t
val div' : t -> float -> t


val sum : t -> float
val mean : t -> float

val reshape : t -> shape:int array -> t
val unsqueeze: t -> axis:int -> t

val transpose : t -> t
val matmul : t -> t -> t
