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
  let grad = Option.None in
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
  if x.requires_grad
  then (
    match x.grad with
    | None -> x.grad <- Some (Ndarray.copy g)
    | Some old -> x.grad <- Some (Ndarray.add old g))
;;

let add a b =
  let requires_grad = a.requires_grad || b.requires_grad in
  let value = Ndarray.add a.value b.value in
  let parents = [ a; b ] in
  let backward g =
    add_grad a g;
    add_grad b g
  in
  { value; parents; backward; grad = Option.None; requires_grad }
;;

let neg a b =
  let value = Ndarray.neg a.value in
  let parents = [ a ] in
  let backward g = add_grad a (Ndarray.neg g) in
  { a with value; parents; backward }
;;

let sub a b =
  let requires_grad = a.requires_grad || b.requires_grad in
  let value = Ndarray.sub a.value b.value in
  let parents = [ a; b ] in
  let backward g =
    add_grad a g;
    add_grad b (Ndarray.neg g)
  in
  { value; parents; backward; grad = Option.None; requires_grad }
;;

let mul a b =
  let requires_grad = a.requires_grad || b.requires_grad in
  let value = Ndarray.mul a.value b.value in
  let parents = [ a; b ] in
  let backward g =
    add_grad a (Ndarray.mul g b.value);
    add_grad b (Ndarray.mul g a.value)
  in
  { value; parents; backward; grad = Option.None; requires_grad }
;;

(* let sum a b = *)
(* let value = Ndarray.sum a.value in *)
(* let parents = [ a ] in *)
(* let backward g = add_grad a (Ndarray.mul (Ndarray.ones_like a.value) g) in *)
(* { a with value; parents; backward } *)
(* ;; *)
