open! Base

type t

val create : int array -> float -> t
val full : int array -> float -> t
val scalar : float -> t
val s : float -> t
val zeros : int array -> t
val ones : int array -> t
val seed : int -> unit
val rand : int array -> t
val randn : int array -> t
val uniform : int array -> low:float -> high:float -> t
val copy : t -> t
val zeros_like : t -> t
val ones_like : t -> t
val of_array : shape:int array -> float array -> t
val to_array : t -> float array
val shape : t -> int array
val ndim : t -> int
val numel : t -> int
val item : t -> float
val get : t -> int array -> float
val set : t -> int array -> float -> unit
val update : t -> int array -> f:(float -> float) -> unit
val map : t -> f:(float -> float) -> t
val map2 : t -> t -> f:(float -> float -> float) -> t
val broadcast_shape : int array -> int array -> int array
val broadcast_to : t -> shape:int array -> t
val add : t -> t -> t
val sub : t -> t -> t
val mul : t -> t -> t
val div : t -> t -> t
val powf : t -> float -> t
val exp : t -> t
val log : t -> t
val tanh : t -> t
val sigmoid : t -> t
val softmax : t -> t
val softmax_axis : t -> axis:int -> t
val maximum : t -> t -> t
val minimum : t -> t -> t
val gt : t -> t -> t
val ge : t -> t -> t
val lt : t -> t -> t
val le : t -> t -> t
val neg : t -> t

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

val sum : t -> t
val mean : t -> t
val max : t -> t
val min : t -> t
val sum_axis : ?keepdim:bool -> t -> axis:int -> t
val mean_axis : ?keepdim:bool -> t -> axis:int -> t
val max_axis : ?keepdim:bool -> t -> axis:int -> t
val min_axis : ?keepdim:bool -> t -> axis:int -> t
val sum_to_shape : t -> shape:int array -> t
val reshape : t -> shape:int array -> t
val unsqueeze : t -> axis:int -> t
val transpose : t -> t
val matmul : t -> t -> t
val arange : int -> t
val linspace : start:float -> stop:float -> num:int -> t
val eye : int -> t
