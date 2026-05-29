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

let add a b =
  let open Ndarray in
  let open Infix in
  let requires_grad = a.requires_grad || b.requires_grad in
  let value = a.value + b.value in
  let parents = [ a; b ] in
  let backward g =
    add_grad a g;
    add_grad b g
  in
  { value; parents; backward; grad = None; requires_grad }
;;

let neg a =
  let open Ndarray in
  let open Infix in
  let value = neg a.value in
  let parents = [ a ] in
  let backward g = add_grad a (neg g) in
  { value
  ; parents
  ; backward
  ; grad = None
  ; requires_grad = a.requires_grad
  }
;;

let relu a =
  let open Ndarray in
  let open Infix in
  let value = maximum a.value (s 0.0) in
  let parents = [ a ] in
  let backward g = add_grad a ((a.value > s 0.0) * g) in
  { value
  ; parents
  ; backward
  ; grad = None
  ; requires_grad = a.requires_grad
  }
;;

let sub a b =
  let open Ndarray in
  let open Infix in
  let requires_grad = a.requires_grad || b.requires_grad in
  let value = a.value - b.value in
  let parents = [ a; b ] in
  let backward g =
    add_grad a g;
    add_grad b (neg g)
  in
  { value; parents; backward; grad = None; requires_grad }
;;

let mul a b =
  let open Ndarray in
  let open Infix in
  let requires_grad = a.requires_grad || b.requires_grad in
  let value = a.value * b.value in
  let parents = [ a; b ] in
  let backward g =
    add_grad a (g * b.value);
    add_grad b (g * a.value)
  in
  { value; parents; backward; grad = None; requires_grad }
;;

let div a b =
  let open Ndarray in
  let open Infix in
  let requires_grad = a.requires_grad || b.requires_grad in
  let value = a.value / b.value in
  let parents = [ a; b ] in
  let backward g =
    add_grad a (g / b.value);
    add_grad b (s (-1.) * g * a.value / (b.value * b.value))
  in
  { value; parents; backward; grad = None; requires_grad }
;;

let matmul a b =
  let open Ndarray in
  let open Infix in
  let requires_grad = a.requires_grad || b.requires_grad in
  let value = a.value @ b.value in
  let parents = [ a; b ] in
  let backward g =
    add_grad a (g @ transpose b.value);
    add_grad b (transpose a.value @ g)
  in
  { value; parents; backward; grad = None; requires_grad }
;;

let sum a =
  let open Ndarray in
  let open Infix in
  let value = sum a.value in
  let parents = [ a ] in
  let backward g = add_grad a (broadcast_to g ~shape:(shape a.value)) in
  { value
  ; parents
  ; backward
  ; grad = None
  ; requires_grad = a.requires_grad
  }
;;

let mean a =
  let open Ndarray in
  let open Infix in
  let value = mean a.value in
  let parents = [ a ] in
  let backward g =
    let n = numel a.value |> Float.of_int |> s in
    let g = broadcast_to g ~shape:(shape a.value) in
    add_grad a (g / n)
  in
  { value
  ; parents
  ; backward
  ; grad = None
  ; requires_grad = a.requires_grad
  }
;;

let powf a p =
  let open Ndarray in
  let open Infix in
  let value = a.value ^ p in
  let parents = [ a ] in
  let backward g =
    let da = s p * (a.value ^ (p -. 1.)) in
    add_grad a (da * g)
  in
  { value
  ; parents
  ; backward
  ; grad = None
  ; requires_grad = a.requires_grad
  }
;;

let square x = powf x 2.

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

let sum_axis ?(keepdim = false) x ~axis = x
let mean_axis ?(keepdim = false) x ~axis = x
