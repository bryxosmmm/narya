open! Base

type t

val create : ?requires_grad:bool -> shape:int array -> float -> t
val zeros : ?requires_grad:bool -> int array -> t
val ones : ?requires_grad:bool -> int array -> t
val scalar : ?requires_grad:bool -> float -> t
val of_ndarray : ?requires_grad:bool -> Ndarray.t -> t
val value : t -> Ndarray.t
val item : t -> float
val grad : t -> Ndarray.t option
val grad_exn : t -> Ndarray.t
val requires_grad : t -> bool
val op : t -> string
val zero_grad : t -> unit
val update : t -> f:(Ndarray.t -> Ndarray.t) -> unit
val add : t -> t -> t
val sub : t -> t -> t
val mul : t -> t -> t
val div : t -> t -> t
val matmul : t -> t -> t
val transpose : t -> t
val neg : t -> t
val square : t -> t
val powf : t -> float -> t
val sqrt : t -> t
val abs : t -> t
val exp : t -> t
val log : t -> t
val tanh : t -> t
val sigmoid : t -> t
val softmax : t -> t
val softmax_axis : t -> axis:int -> t
val gt : t -> t -> t
val ge : t -> t -> t
val lt : t -> t -> t
val le : t -> t -> t
val relu : t -> t
val sum : t -> t
val mean : t -> t
val sum_axis : ?keepdim:bool -> t -> axis:int -> t
val mean_axis : ?keepdim:bool -> t -> axis:int -> t

module Infix : sig
  val ( + ) : t -> t -> t
  val ( - ) : t -> t -> t
  val ( * ) : t -> t -> t
  val ( / ) : t -> t -> t
  val ( ^ ) : t -> float -> t
  val ( > ) : t -> t -> t
  val ( >= ) : t -> t -> t
  val ( < ) : t -> t -> t
  val ( <= ) : t -> t -> t
  val ( ~- ) : t -> t
  val ( @ ) : t -> t -> t
end

val backward : t -> unit

module Debug : sig
  val to_dot : t -> string
end
