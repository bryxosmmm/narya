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
let to_array x = Array.copy x.data
let same_shape = Array.equal Int.equal
let get2 x i j = x.data.((i * x.strides.(0)) + (j * x.strides.(1)) + x.offset)

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

let create shape v =
  let shape = Array.copy shape in
  let ndim = Array.length shape in
  let numel = comp_numel shape in
  let strides = comp_strides shape in
  let offset = 0 in
  let data = Array.init numel ~f:(fun _ -> v) in
  { data; ndim; numel; strides; offset; shape }
;;

let zeros shape = create shape 0.0
let ones shape = create shape 1.0

let index x idx =
  let offset = x.offset in
  let s = x.strides in
  let n = x.ndim in
  let rec aux i = if i >= n then offset else (s.(i) * idx.(i)) + aux (i + 1) in
  aux 0
;;

let get x idx = x.data.(index x idx)
let set x idx v = x.data.(index x idx) <- v

(* val map2 : t -> t -> f:(float -> float -> float) -> t *)

let update x idx ~f =
  let i = index x idx in
  x.data.(i) <- f x.data.(i)
;;

let map x ~f = { x with data = Array.map x.data ~f }

let map2 x y ~f =
  if not (same_shape x.shape y.shape) then invalid_arg "Ndarray.map2: shape mismatch";
  { x with data = Array.map2_exn x.data y.data ~f }
;;

let unsqueeze x ~axis =
  let ndim' = x.ndim + 1 in
  let shape' =
    Array.init ndim' ~f:(fun i ->
      if i < axis then x.shape.(i) else if i = axis then 1 else x.shape.(i - 1))
  in
  let strides' = comp_strides shape' in
  { x with shape = shape'; ndim = ndim'; strides = strides' }
;;

let reshape x ~shape =
  let shape' = Array.copy shape in
  let numel' = comp_numel shape' in
  if x.numel <> numel' then invalid_arg "Ndarray.reshape: shape mismatch";
  let strides' = comp_strides shape' in
  let ndim' = Array.length shape' in
  { x with numel = numel'; shape = shape'; strides = strides'; ndim = ndim' }
;;

let add a b = map2 a b ~f:( +. )
let sub a b = map2 a b ~f:( -. )
let mul a b = map2 a b ~f:( *. )
let div a b = map2 a b ~f:( /. )
let add' a s = map a ~f:(fun x -> x +. s)
let sub' a s = map a ~f:(fun x -> x -. s)
let mul' a s = map a ~f:(fun x -> x *. s)
let div' a s = map a ~f:(fun x -> x /. s)
let neg a = map a ~f:Float.neg

(* val sum : t -> float *)
(* val mean : t -> float *)
let sum x = Array.fold x.data ~init:0.0 ~f:( +. )

let mean x =
  if x.numel = 0
  then invalid_arg "Ndarray.mean: empty array"
  else sum x /. Float.of_int x.numel
;;

let transpose x =
  if x.ndim <> 2 then invalid_arg "Ndarray.transpose: number of dimensions <> 2";
  let n = x.shape.(0) in
  let m = x.shape.(1) in
  let numel = n * m in
  let shape = [| m; n |] in
  let strides = comp_strides shape in
  let data =
    Array.init numel ~f:(fun flat ->
      let i = flat / n in
      let j = flat % n in
      get2 x j i)
  in
  { data; strides; shape; numel; ndim = 2; offset = 0 }
;;

let matmul a b =
  if a.ndim <> 2 then invalid_arg "Ndarray.matmul: left argument must be 2D";
  if b.ndim <> 2 then invalid_arg "Ndarray.matmul: right argument must be 2D";
  if a.shape.(1) <> b.shape.(0) then invalid_arg "Ndarray.matmul: shape mismatch";
  let m = a.shape.(0) in
  let k = a.shape.(1) in
  let n = b.shape.(1) in
  let shape = [| m; n |] in
  let strides = comp_strides shape in
  let numel = m * n in
  let data =
    Array.init numel ~f:(fun flat ->
      let i = flat / n in
      let j = flat % n in
      let rec aux p acc =
        if p >= k then acc else aux (p + 1) (acc +. (get2 a i p *. get2 b p j))
      in
      aux 0 0.0)
  in
  { data; strides; numel; shape; offset = 0; ndim = 2 }
;;

let of_array ~shape data =
  let numel = comp_numel shape in
  if numel <> Array.length data then invalid_arg "Ndarray.of_array: shape mismatch";
  let strides = comp_strides shape in
  let ndim = Array.length shape in
  { data; shape; strides; ndim; numel; offset = 0 }
;;
