open! Base

type t =
  { value : Ndarray.t
  ; mutable grad : Ndarray.t option
  ; requires_grad : bool
  ; backward : Ndarray.t -> unit
  ; parents : t list
  }

let value x = x.value
let grad x = x.grad
let grad_exn x = Option.value_exn x.grad
let requires_grad x = x.requires_grad

let of_ndarray ?(requires_grad = false) value =
  let grad = None in
  { value; grad; requires_grad; backward = (fun _ -> ()); parents = [] }
;;

let create ?(requires_grad = false) ~shape v =
  let value = Ndarray.create shape v in
  of_ndarray ~requires_grad value
;;

let zeros ?(requires_grad = false) shape = create ~requires_grad ~shape 0.0
let ones ?(requires_grad = false) shape = create ~requires_grad ~shape 1.0
let scalar ?(requires_grad = false) = create ~requires_grad ~shape:[||]

let add_grad x g =
  let open Ndarray in
  let open Infix in
  if x.requires_grad
  then (
    let g = sum_to_shape g ~shape:(shape x.value) in
    match x.grad with
    | None -> x.grad <- Some (copy g)
    | Some old -> x.grad <- Some (old + g))
;;

let op'1 x ~f ~f' =
  let value = f x.value in
  let parents = [ x ] in
  let backward g = add_grad x (f' x.value value g) in
  { value
  ; parents
  ; backward
  ; grad = None
  ; requires_grad = x.requires_grad
  }
;;

let op'2 a b ~f ~f'a ~f'b =
  let value = f a.value b.value in
  let parents = [ a; b ] in
  let backward g =
    add_grad a (f'a a.value b.value value g);
    add_grad b (f'b a.value b.value value g)
  in
  { value
  ; parents
  ; backward
  ; grad = None
  ; requires_grad = a.requires_grad || b.requires_grad
  }
;;

let add a b =
  let open Ndarray in
  let open Infix in
  op'2 a b ~f:( + ) ~f'a:(fun _ _ _ g -> g) ~f'b:(fun _ _ _ g -> g)
;;

let neg a =
  let open Ndarray in
  op'1 a ~f:neg ~f':(fun _ _ g -> neg g)
;;

let relu a =
  let open Ndarray in
  let open Infix in
  op'1 a ~f:(fun x -> maximum x (s 0.0)) ~f':(fun x _ g -> (x > s 0.0) * g)
;;

let sub a b =
  let open Ndarray in
  let open Infix in
  op'2 a b ~f:( - ) ~f'a:(fun _ _ _ g -> g) ~f'b:(fun _ _ _ g -> neg g)
;;

let mul a b =
  let open Ndarray in
  let open Infix in
  op'2 a b ~f:( * ) ~f'a:(fun _ b _ g -> g * b) ~f'b:(fun a _ _ g -> g * a)
;;

let div a b =
  let open Ndarray in
  let open Infix in
  op'2
    a
    b
    ~f:( / )
    ~f'a:(fun _ b _ g -> g / b)
    ~f'b:(fun a b _ g -> s (-1.) * g * a / (b * b))
;;

let matmul a b =
  let open Ndarray in
  let open Infix in
  op'2
    a
    b
    ~f:( @ )
    ~f'a:(fun _ b _ g -> g @ transpose b)
    ~f'b:(fun a _ _ g -> transpose a @ g)
;;

let sum a =
  let open Ndarray in
  op'1 a ~f:sum ~f':(fun x _ g -> broadcast_to g ~shape:(shape x))
;;

let mean a =
  let open Ndarray in
  let open Infix in
  op'1 a ~f:mean ~f':(fun x _ g ->
    let n = numel x |> Float.of_int |> s in
    broadcast_to g ~shape:(shape x) / n)
;;

let powf a p =
  let open Ndarray in
  let open Infix in
  op'1 a ~f:(fun x -> x ^ p) ~f':(fun x _ g -> g * s p * (x ^ (p -. 1.)))
;;

let square x = powf x 2.

let exp x =
  let open Ndarray in
  let open Infix in
  op'1 x ~f:exp ~f':(fun _ v g -> g * v)
;;

let log x =
  let open Ndarray in
  let open Infix in
  op'1 x ~f:log ~f':(fun x _ g -> g / x)
;;

let tanh x =
  let open Ndarray in
  let open Infix in
  op'1 x ~f:tanh ~f':(fun _ v g -> g * (s 1. - powf v 2.))
;;

let sigmoid x =
  let open Ndarray in
  let open Infix in
  op'1 x ~f:sigmoid ~f':(fun _ v g -> g * v * (s 1. - v))
;;

let topo_sort root =
  let vis = ref [] in
  let t = ref [] in
  let rec dfs u =
    if not (List.exists !vis ~f:(phys_equal u))
    then (
      vis := u :: !vis;
      List.iter u.parents ~f:dfs;
      t := u :: !t)
  in
  dfs root;
  !t
;;

let backward x =
  x.grad <- Some (Ndarray.ones_like x.value);
  List.iter (topo_sort x) ~f:(fun v -> Option.iter v.grad ~f:v.backward)
;;

let sum_axis ?(keepdim = false) a ~axis =
  let open Ndarray in
  op'1
    a
    ~f:(fun x -> sum_axis ~keepdim x ~axis)
    ~f':(fun x _ g ->
      let g = if keepdim then g else unsqueeze g ~axis in
      broadcast_to g ~shape:(shape x))
;;

let mean_axis ?(keepdim = false) a ~axis =
  let open Ndarray in
  let open Infix in
  op'1
    a
    ~f:(fun x -> mean_axis ~keepdim x ~axis)
    ~f':(fun x _ g ->
      let n = (shape x).(axis) |> Float.of_int |> s in
      let g = if keepdim then g else unsqueeze g ~axis in
      broadcast_to g ~shape:(shape x) / n)
;;
