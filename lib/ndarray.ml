open! Base

type t =
  { data : float array
  ; shape : int array
  ; strides : int array
  ; offset : int
  ; ndim : int
  ; numel : int
  }

let shape x = Array.copy x.shape
let ndim x = x.ndim
let numel x = x.numel
let same_shape = Array.equal Int.equal

let comp_numel shape =
  let l = Array.length shape in
  let rec aux = function
    | 0 -> shape.(0)
    | n -> shape.(n) * aux (n - 1)
  in
  if l = 0 then 1 else aux (l - 1)
;;

let comp_strides shape =
  let n = Array.length shape in
  let rec aux i cur acc =
    if i < 0 then acc else aux (i - 1) (cur * shape.(i)) (cur :: acc)
  in
  aux (n - 1) 1 [] |> Array.of_list
;;

let make_unchecked ~data ~shape ~strides ~offset =
  { data
  ; shape
  ; strides
  ; offset
  ; ndim = Array.length shape
  ; numel = comp_numel shape
  }
;;

let make_contiguous_unchecked ~shape ~data =
  make_unchecked ~data ~shape ~strides:(comp_strides shape) ~offset:0
;;

let unravel_index shape flat =
  let ndim = Array.length shape in
  let idx = Array.create ~len:ndim 0 in
  let rest = ref flat in
  for dim = ndim - 1 downto 0 do
    idx.(dim) <- !rest % shape.(dim);
    rest := !rest / shape.(dim)
  done;
  idx
;;

let flat_index shape idx =
  let strides = comp_strides shape in
  Array.fold2_exn idx strides ~init:0 ~f:(fun acc i stride ->
    acc + (i * stride))
;;

let insert_axis idx ~axis ~value ~ndim =
  Array.init ndim ~f:(fun i ->
    if i < axis then idx.(i) else if i = axis then value else idx.(i - 1))
;;

let reduced_shape shape ~axis ~keepdim =
  let ndim = Array.length shape in
  if axis < 0 || axis >= ndim
  then invalid_arg "Ndarray.sum_axis: axis out of bounds";
  if keepdim
  then Array.mapi shape ~f:(fun i dim -> if i = axis then 1 else dim)
  else
    Array.init (ndim - 1) ~f:(fun i ->
      if i < axis then shape.(i) else shape.(i + 1))
;;

let create shape v =
  let shape = Array.copy shape in
  let data = Array.init (comp_numel shape) ~f:(fun _ -> v) in
  make_contiguous_unchecked ~shape ~data
;;

let full = create
let scalar v = create [||] v
let s = scalar
let zeros shape = create shape 0.0
let ones shape = create shape 1.0

let of_array ~shape data =
  let shape = Array.copy shape in
  let data = Array.copy data in
  let numel = comp_numel shape in
  if numel <> Array.length data
  then invalid_arg "Ndarray.of_array: shape mismatch";
  make_contiguous_unchecked ~shape ~data
;;

let arange n =
  let shape = [| n |] in
  let data = Array.init n ~f:Float.of_int in
  make_contiguous_unchecked ~shape ~data
;;

let linspace ~start ~stop ~num =
  if num <= 0 then invalid_arg "Ndarray.linspace: num > 0";
  let data =
    if num = 1
    then [| start |]
    else (
      let v = (stop -. start) /. (Float.of_int num -. 1.) in
      Array.init num ~f:(fun i ->
        if i = num - 1 then stop else start +. (Float.of_int i *. v)))
  in
  make_contiguous_unchecked ~shape:[| num |] ~data
;;

let eye n =
  let shape = [| n; n |] in
  let data =
    Array.init (n * n) ~f:(fun i ->
      (if i % (n + 1) = 0 then 1 else 0) |> Float.of_int)
  in
  make_contiguous_unchecked ~shape ~data
;;

let index x idx =
  let offset = x.offset in
  let s = x.strides in
  let n = x.ndim in
  let rec aux i =
    if i >= n then offset else (s.(i) * idx.(i)) + aux (i + 1)
  in
  aux 0
;;

let index_flat x flat =
  let rec aux dim flat acc =
    if dim < 0
    then acc
    else (
      let i = flat % x.shape.(dim) in
      aux (dim - 1) (flat / x.shape.(dim)) (acc + (i * x.strides.(dim))))
  in
  aux (x.ndim - 1) flat x.offset
;;

let get x idx = x.data.(index x idx)
let set x idx v = x.data.(index x idx) <- v
let get_flat x flat = x.data.(index_flat x flat)

let get2 x i j =
  x.data.((i * x.strides.(0)) + (j * x.strides.(1)) + x.offset)
;;

let update x idx ~f =
  let i = index x idx in
  x.data.(i) <- f x.data.(i)
;;

let to_array x = Array.init x.numel ~f:(fun i -> get_flat x i)
let is_contiguous x = same_shape x.strides (comp_strides x.shape)

let copy x =
  make_contiguous_unchecked ~shape:(Array.copy x.shape) ~data:(to_array x)
;;

let like x v =
  make_contiguous_unchecked
    ~shape:(Array.copy x.shape)
    ~data:(Array.init x.numel ~f:(fun _ -> v))
;;

let zeros_like x = like x 0.0
let ones_like x = like x 1.0

let reshape x ~shape =
  if not (is_contiguous x)
  then invalid_arg "Ndarray.reshape: non-contiguous view";
  let shape = Array.copy shape in
  let numel = comp_numel shape in
  if x.numel <> numel then invalid_arg "Ndarray.reshape: shape mismatch";
  { x with
    numel
  ; shape
  ; strides = comp_strides shape
  ; ndim = Array.length shape
  }
;;

let unsqueeze x ~axis =
  if axis < 0 || axis > x.ndim
  then invalid_arg "Ndarray.unsqueeze: axis out of bounds";
  let ndim = x.ndim + 1 in
  let shape =
    Array.init ndim ~f:(fun i ->
      if i < axis
      then x.shape.(i)
      else if i = axis
      then 1
      else x.shape.(i - 1))
  in
  let strides =
    Array.init ndim ~f:(fun i ->
      if i < axis
      then x.strides.(i)
      else if i = axis
      then 0
      else x.strides.(i - 1))
  in
  { x with shape; ndim; strides; numel = x.numel }
;;

let transpose x =
  if x.ndim <> 2
  then invalid_arg "Ndarray.transpose: number of dimensions <> 2";
  { x with
    shape = [| x.shape.(1); x.shape.(0) |]
  ; strides = [| x.strides.(1); x.strides.(0) |]
  }
;;

let broadcast_shape a b =
  let na = Array.length a in
  let nb = Array.length b in
  let n = Int.max na nb in
  Array.init n ~f:(fun i ->
    let ai = i - (n - na) in
    let bi = i - (n - nb) in
    let ad = if ai < 0 then 1 else a.(ai) in
    let bd = if bi < 0 then 1 else b.(bi) in
    if ad = bd
    then ad
    else if ad = 1
    then bd
    else if bd = 1
    then ad
    else invalid_arg "Ndarray.broadcast_shape: incompatible shapes")
;;

let broadcast_to x ~shape =
  let shape = Array.copy shape in
  let ndim = Array.length shape in
  if ndim < x.ndim
  then invalid_arg "Ndarray.broadcast_to: target rank too small";
  let shift = ndim - x.ndim in
  let strides =
    Array.init ndim ~f:(fun i ->
      let xi = i - shift in
      if xi < 0
      then 0
      else if x.shape.(xi) = shape.(i)
      then x.strides.(xi)
      else if x.shape.(xi) = 1
      then 0
      else invalid_arg "Ndarray.broadcast_to: incompatible shape")
  in
  { x with shape; strides; ndim; numel = comp_numel shape }
;;

let map x ~f =
  make_contiguous_unchecked
    ~shape:(Array.copy x.shape)
    ~data:(Array.init x.numel ~f:(fun i -> f (get_flat x i)))
;;

let map2 x y ~f =
  let shape = broadcast_shape x.shape y.shape in
  let x = broadcast_to x ~shape in
  let y = broadcast_to y ~shape in
  let data =
    Array.init (comp_numel shape) ~f:(fun i ->
      f (get_flat x i) (get_flat y i))
  in
  make_contiguous_unchecked ~shape ~data
;;

let add a b = map2 a b ~f:( +. )
let sub a b = map2 a b ~f:( -. )
let mul a b = map2 a b ~f:( *. )
let div a b = map2 a b ~f:( /. )
let powf a p = map a ~f:(fun x -> Float.(x ** p))
let exp a = map a ~f:Float.exp
let log a = map a ~f:Float.log
let tanh a = map a ~f:Float.tanh
let sigmoid a = map a ~f:(fun x -> 1. /. (1. +. Float.exp (-.x)))
let maximum a b = map2 a b ~f:Float.max
let minimum a b = map2 a b ~f:Float.min
let gt a b = map2 a b ~f:(fun x y -> if Float.(x > y) then 1.0 else 0.0)
let ge a b = map2 a b ~f:(fun x y -> if Float.(x >= y) then 1.0 else 0.0)
let lt a b = map2 a b ~f:(fun x y -> if Float.(x < y) then 1.0 else 0.0)
let le a b = map2 a b ~f:(fun x y -> if Float.(x <= y) then 1.0 else 0.0)
let neg a = map a ~f:Float.neg
let sum x = Array.fold (to_array x) ~init:0.0 ~f:( +. ) |> scalar

let max x =
  if x.numel = 0
  then invalid_arg "Ndarray.max: empty array"
  else Array.reduce_exn (to_array x) ~f:Float.max |> scalar
;;

let min x =
  if x.numel = 0
  then invalid_arg "Ndarray.min: empty array"
  else Array.reduce_exn (to_array x) ~f:Float.min |> scalar
;;

let mean x =
  if x.numel = 0
  then invalid_arg "Ndarray.mean: empty array"
  else div (sum x) (scalar (Float.of_int x.numel))
;;

let softmax x =
  let e = exp (sub x (max x)) in
  div e (sum e)
;;

let sum_axis ?(keepdim = false) x ~axis =
  let shape = reduced_shape x.shape ~axis ~keepdim in
  let numel = comp_numel shape in
  let data =
    Array.init numel ~f:(fun flat ->
      let out_idx = unravel_index shape flat in
      let out_idx_no_keepdim =
        if keepdim
        then
          Array.init (x.ndim - 1) ~f:(fun i ->
            if i < axis then out_idx.(i) else out_idx.(i + 1))
        else out_idx
      in
      let acc = ref 0.0 in
      for i = 0 to x.shape.(axis) - 1 do
        let idx =
          insert_axis out_idx_no_keepdim ~axis ~value:i ~ndim:x.ndim
        in
        acc := !acc +. get x idx
      done;
      !acc)
  in
  make_contiguous_unchecked ~shape ~data
;;

let mean_axis ?(keepdim = false) x ~axis =
  if axis < 0 || axis >= x.ndim
  then invalid_arg "Ndarray.mean_axis: axis out of bounds";
  div (sum_axis x ~axis ~keepdim) (scalar (Float.of_int x.shape.(axis)))
;;

let sum_to_shape x ~shape =
  let shape = Array.copy shape in
  let ndim = Array.length shape in
  if ndim > x.ndim
  then invalid_arg "Ndarray.sum_to_shape: target rank too large";
  let shift = x.ndim - ndim in
  for i = 0 to ndim - 1 do
    let x_dim = x.shape.(i + shift) in
    let target_dim = shape.(i) in
    if target_dim <> x_dim && target_dim <> 1
    then invalid_arg "Ndarray.sum_to_shape: incompatible shape"
  done;
  let data = Array.create ~len:(comp_numel shape) 0.0 in
  for flat = 0 to x.numel - 1 do
    let x_idx = unravel_index x.shape flat in
    let out_idx =
      Array.init ndim ~f:(fun i ->
        if shape.(i) = 1 then 0 else x_idx.(i + shift))
    in
    let out_flat = flat_index shape out_idx in
    data.(out_flat) <- data.(out_flat) +. get x x_idx
  done;
  make_contiguous_unchecked ~shape ~data
;;

let matmul a b =
  if a.ndim <> 2
  then invalid_arg "Ndarray.matmul: left argument must be 2D";
  if b.ndim <> 2
  then invalid_arg "Ndarray.matmul: right argument must be 2D";
  if a.shape.(1) <> b.shape.(0)
  then invalid_arg "Ndarray.matmul: shape mismatch";
  let m = a.shape.(0) in
  let k = a.shape.(1) in
  let n = b.shape.(1) in
  let shape = [| m; n |] in
  let data =
    Array.init (m * n) ~f:(fun flat ->
      let i = flat / n in
      let j = flat % n in
      let rec aux p acc =
        if p >= k
        then acc
        else aux (p + 1) (acc +. (get2 a i p *. get2 b p j))
      in
      aux 0 0.0)
  in
  make_contiguous_unchecked ~shape ~data
;;

module Infix = struct
  let ( + ) = add
  let ( - ) = sub
  let ( * ) = mul
  let ( / ) = div
  let ( ^ ) = powf
  let ( > ) = gt
  let ( >= ) = ge
  let ( < ) = lt
  let ( <= ) = le
  let ( ~- ) = neg
  let ( @ ) = matmul
end
