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
let numel' shape = Array.fold shape ~init:1 ~f:( * )

let strides' shape =
  let rec loop i s acc =
    if i < 0 then acc else loop (i - 1) (s * shape.(i)) (s :: acc)
  in
  loop (Array.length shape - 1) 1 [] |> Array.of_list
;;

let unsafe_view ~data ~shape ~strides ~offset =
  { data
  ; shape
  ; strides
  ; offset
  ; ndim = Array.length shape
  ; numel = numel' shape
  }
;;

let unsafe_make ~shape ~data =
  unsafe_view ~data ~shape ~strides:(strides' shape) ~offset:0
;;

let unravel_index shape flat =
  let n = Array.length shape in
  let idx = Array.create ~len:n 0 in
  let flat = ref flat in
  for i = n - 1 downto 0 do
    idx.(i) <- !flat % shape.(i);
    flat := !flat / shape.(i)
  done;
  idx
;;

let flat_index shape idx =
  Array.fold2_exn idx (strides' shape) ~init:0 ~f:(fun flat i s ->
    flat + (i * s))
;;

let insert_axis idx ~axis ~value ~ndim =
  Array.init ndim ~f:(fun i ->
    if i < axis then idx.(i) else if i = axis then value else idx.(i - 1))
;;

let reduced_shape shape ~axis ~keepdim =
  let n = Array.length shape in
  if axis < 0 || axis >= n
  then invalid_arg "Ndarray.sum_axis: axis out of bounds";
  if keepdim
  then Array.mapi shape ~f:(fun i d -> if i = axis then 1 else d)
  else
    Array.init (n - 1) ~f:(fun i ->
      if i < axis then shape.(i) else shape.(i + 1))
;;

let create shape v =
  let shape = Array.copy shape in
  let data = Array.init (numel' shape) ~f:(fun _ -> v) in
  unsafe_make ~shape ~data
;;

let full = create
let scalar v = create [||] v
let s = scalar
let zeros shape = create shape 0.0
let ones shape = create shape 1.0
let seed = Stdlib.Random.init

let rand shape =
  let shape = Array.copy shape in
  let data = Array.init (numel' shape) ~f:(fun _ -> Stdlib.Random.float 1.0) in
  unsafe_make ~shape ~data
;;

let uniform shape ~low ~high =
  let shape = Array.copy shape in
  let data =
    Array.init (numel' shape) ~f:(fun _ -> low +. (Stdlib.Random.float 1.0 *. (high -. low)))
  in
  unsafe_make ~shape ~data
;;

let randn shape =
  let shape = Array.copy shape in
  let data =
    Array.init (numel' shape) ~f:(fun _ ->
      let u1 = Float.max Float.epsilon_float (Stdlib.Random.float 1.0) in
      let u2 = Stdlib.Random.float 1.0 in
      Float.sqrt (-2.0 *. Float.log u1) *. Float.cos (2.0 *. Float.pi *. u2))
  in
  unsafe_make ~shape ~data
;;

let of_array ~shape data =
  let shape = Array.copy shape in
  let data = Array.copy data in
  let numel = numel' shape in
  if numel <> Array.length data
  then invalid_arg "Ndarray.of_array: shape mismatch";
  unsafe_make ~shape ~data
;;

let arange n =
  let shape = [| n |] in
  let data = Array.init n ~f:Float.of_int in
  unsafe_make ~shape ~data
;;

let linspace ~start ~stop ~num =
  if num <= 0 then invalid_arg "Ndarray.linspace: num > 0";
  let data =
    if num = 1
    then [| start |]
    else (
      let step = (stop -. start) /. (Float.of_int num -. 1.) in
      Array.init num ~f:(fun i ->
        if i = num - 1 then stop else start +. (Float.of_int i *. step)))
  in
  unsafe_make ~shape:[| num |] ~data
;;

let eye n =
  let shape = [| n; n |] in
  let data =
    Array.init (n * n) ~f:(fun i ->
      (if i % (n + 1) = 0 then 1 else 0) |> Float.of_int)
  in
  unsafe_make ~shape ~data
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

let item x =
  if x.numel <> 1 then invalid_arg "Ndarray.item: expected exactly one element";
  get_flat x 0
;;

let get2 x i j =
  x.data.((i * x.strides.(0)) + (j * x.strides.(1)) + x.offset)
;;

let update x idx ~f =
  let i = index x idx in
  x.data.(i) <- f x.data.(i)
;;

let to_array x = Array.init x.numel ~f:(fun i -> get_flat x i)
let is_contiguous x = same_shape x.strides (strides' x.shape)
let copy x = unsafe_make ~shape:(Array.copy x.shape) ~data:(to_array x)

let like x v =
  unsafe_make
    ~shape:(Array.copy x.shape)
    ~data:(Array.init x.numel ~f:(fun _ -> v))
;;

let zeros_like x = like x 0.0
let ones_like x = like x 1.0

let reshape x ~shape =
  if not (is_contiguous x)
  then invalid_arg "Ndarray.reshape: non-contiguous view";
  let shape = Array.copy shape in
  let numel = numel' shape in
  if x.numel <> numel then invalid_arg "Ndarray.reshape: shape mismatch";
  { x with
    numel
  ; shape
  ; strides = strides' shape
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
  { x with shape; strides; ndim; numel = numel' shape }
;;

let map x ~f =
  unsafe_make
    ~shape:(Array.copy x.shape)
    ~data:(Array.init x.numel ~f:(fun i -> f (get_flat x i)))
;;

let map2 x y ~f =
  let shape = broadcast_shape x.shape y.shape in
  let x = broadcast_to x ~shape in
  let y = broadcast_to y ~shape in
  let data =
    Array.init (numel' shape) ~f:(fun i -> f (get_flat x i) (get_flat y i))
  in
  unsafe_make ~shape ~data
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

let reduce_axis ?(keepdim = false) x ~axis ~init ~f =
  let shape = reduced_shape x.shape ~axis ~keepdim in
  let data =
    Array.init (numel' shape) ~f:(fun flat ->
      let idx = unravel_index shape flat in
      let idx' =
        if not keepdim
        then idx
        else
          Array.init (x.ndim - 1) ~f:(fun i ->
            if i < axis then idx.(i) else idx.(i + 1))
      in
      let acc = ref init in
      for i = 0 to x.shape.(axis) - 1 do
        acc
        := f !acc (get x (insert_axis idx' ~axis ~value:i ~ndim:x.ndim))
      done;
      !acc)
  in
  unsafe_make ~shape ~data
;;

let sum_axis ?(keepdim = false) x ~axis =
  reduce_axis x ~axis ~keepdim ~init:0.0 ~f:( +. )
;;

let mean_axis ?(keepdim = false) x ~axis =
  if axis < 0 || axis >= x.ndim
  then invalid_arg "Ndarray.mean_axis: axis out of bounds";
  div (sum_axis x ~axis ~keepdim) (scalar (Float.of_int x.shape.(axis)))
;;

let max_axis ?(keepdim = false) x ~axis =
  if x.shape.(axis) = 0 then invalid_arg "Ndarray.max_axis: empty axis";
  reduce_axis x ~axis ~keepdim ~init:Float.neg_infinity ~f:Float.max
;;

let min_axis ?(keepdim = false) x ~axis =
  if x.shape.(axis) = 0 then invalid_arg "Ndarray.min_axis: empty axis";
  reduce_axis x ~axis ~keepdim ~init:Float.infinity ~f:Float.min
;;

let softmax_axis x ~axis =
  let m = max_axis ~keepdim:true x ~axis in
  let e = exp (sub x m) in
  let s = sum_axis e ~keepdim:true ~axis in
  div e s
;;

let sum_to_shape x ~shape =
  let shape = Array.copy shape in
  let n = Array.length shape in
  if n > x.ndim
  then invalid_arg "Ndarray.sum_to_shape: target rank too large";
  let d = x.ndim - n in
  for i = 0 to n - 1 do
    let a = x.shape.(i + d) in
    let b = shape.(i) in
    if a <> b && b <> 1
    then invalid_arg "Ndarray.sum_to_shape: incompatible shape"
  done;
  let data = Array.create ~len:(numel' shape) 0.0 in
  for flat = 0 to x.numel - 1 do
    let idx = unravel_index x.shape flat in
    let idx' =
      Array.init n ~f:(fun i -> if shape.(i) = 1 then 0 else idx.(i + d))
    in
    let flat' = flat_index shape idx' in
    data.(flat') <- data.(flat') +. get x idx
  done;
  unsafe_make ~shape ~data
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
  unsafe_make ~shape ~data
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
