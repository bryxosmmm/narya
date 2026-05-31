open! Base

type t =
  { mutable value : Ndarray.t
  ; mutable grad : Ndarray.t option
  ; requires_grad : bool
  ; backward : Ndarray.t -> unit
  ; parents : t list
  ; op : string
  }

let create_node ~value ~parents ~requires_grad ~backward ~op =
  { value; grad = None; requires_grad; backward; parents; op }
;;

let value x = x.value
let grad x = x.grad
let grad_exn x = Option.value_exn x.grad
let requires_grad x = x.requires_grad
let op x = x.op
let zero_grad x = x.grad <- None
let update x ~f = x.value <- f x.value

let of_ndarray ?(requires_grad = false) value =
  create_node
    ~value
    ~parents:[]
    ~requires_grad
    ~backward:(fun _ -> ())
    ~op:"leaf"
;;

let create ?(requires_grad = false) ~shape v =
  let value = Ndarray.create shape v in
  of_ndarray ~requires_grad value
;;

let zeros ?(requires_grad = false) shape = create ~requires_grad ~shape 0.0
let ones ?(requires_grad = false) shape = create ~requires_grad ~shape 1.0
let scalar ?(requires_grad = false) = create ~requires_grad ~shape:[||]

module Node = struct
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

  let unary x ~op ~f ~df =
    let value = f x.value in
    let parents = [ x ] in
    let backward g = add_grad x (df x.value value g) in
    create_node
      ~value
      ~parents
      ~backward
      ~op
      ~requires_grad:x.requires_grad
  ;;

  let binary a b ~op ~f ~fa ~fb =
    let value = f a.value b.value in
    let parents = [ a; b ] in
    let backward g =
      add_grad a (fa a.value b.value value g);
      add_grad b (fb a.value b.value value g)
    in
    create_node
      ~value
      ~parents
      ~backward
      ~op
      ~requires_grad:(a.requires_grad || b.requires_grad)
  ;;
end

module Autograd = struct
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
end

let add a b =
  let open Ndarray in
  let open Infix in
  Node.binary
    a
    b
    ~op:"add"
    ~f:( + )
    ~fa:(fun _ _ _ g -> g)
    ~fb:(fun _ _ _ g -> g)
;;

let sub a b =
  let open Ndarray in
  let open Infix in
  Node.binary
    a
    b
    ~op:"sub"
    ~f:( - )
    ~fa:(fun _ _ _ g -> g)
    ~fb:(fun _ _ _ g -> neg g)
;;

let mul a b =
  let open Ndarray in
  let open Infix in
  Node.binary
    a
    b
    ~op:"mul"
    ~f:( * )
    ~fa:(fun _ b _ g -> g * b)
    ~fb:(fun a _ _ g -> g * a)
;;

let div a b =
  let open Ndarray in
  let open Infix in
  Node.binary
    a
    b
    ~op:"div"
    ~f:( / )
    ~fa:(fun _ b _ g -> g / b)
    ~fb:(fun a b _ g -> s (-1.) * g * a / (b * b))
;;

let neg a =
  let open Ndarray in
  Node.unary a ~op:"neg" ~f:neg ~df:(fun _ _ g -> neg g)
;;

let matmul a b =
  let open Ndarray in
  let open Infix in
  Node.binary
    a
    b
    ~op:"matmul"
    ~f:( @ )
    ~fa:(fun _ b _ g -> g @ transpose b)
    ~fb:(fun a _ _ g -> transpose a @ g)
;;

let sum a =
  let open Ndarray in
  Node.unary a ~op:"sum" ~f:sum ~df:(fun x _ g ->
    broadcast_to g ~shape:(shape x))
;;

let mean a =
  let open Ndarray in
  let open Infix in
  Node.unary a ~op:"mean" ~f:mean ~df:(fun x _ g ->
    let n = numel x |> Float.of_int |> s in
    broadcast_to g ~shape:(shape x) / n)
;;

let sum_axis ?(keepdim = false) a ~axis =
  let open Ndarray in
  Node.unary
    a
    ~op:"sum_axis"
    ~f:(fun x -> sum_axis ~keepdim x ~axis)
    ~df:(fun x _ g ->
      let g = if keepdim then g else unsqueeze g ~axis in
      broadcast_to g ~shape:(shape x))
;;

let mean_axis ?(keepdim = false) a ~axis =
  let open Ndarray in
  let open Infix in
  Node.unary
    a
    ~op:"mean_axis"
    ~f:(fun x -> mean_axis ~keepdim x ~axis)
    ~df:(fun x _ g ->
      let n = (shape x).(axis) |> Float.of_int |> s in
      let g = if keepdim then g else unsqueeze g ~axis in
      broadcast_to g ~shape:(shape x) / n)
;;

let powf a p =
  let open Ndarray in
  let open Infix in
  Node.unary
    a
    ~op:"powf"
    ~f:(fun x -> x ^ p)
    ~df:(fun x _ g -> g * s p * (x ^ (p -. 1.)))
;;

let square x = powf x 2.

let exp x =
  let open Ndarray in
  let open Infix in
  Node.unary x ~op:"exp" ~f:exp ~df:(fun _ v g -> g * v)
;;

let log x =
  let open Ndarray in
  let open Infix in
  Node.unary x ~op:"log" ~f:log ~df:(fun x _ g -> g / x)
;;

let tanh x =
  let open Ndarray in
  let open Infix in
  Node.unary x ~op:"tanh" ~f:tanh ~df:(fun _ v g -> g * (s 1. - powf v 2.))
;;

let sigmoid x =
  let open Ndarray in
  let open Infix in
  Node.unary x ~op:"sigmoid" ~f:sigmoid ~df:(fun _ v g ->
    g * v * (s 1. - v))
;;

let relu a =
  let open Ndarray in
  let open Infix in
  Node.unary
    a
    ~op:"relu"
    ~f:(fun x -> maximum x (s 0.0))
    ~df:(fun x _ g -> (x > s 0.0) * g)
;;

let softmax x =
  let open Ndarray in
  let open Infix in
  Node.unary x ~op:"softmax" ~f:softmax ~df:(fun _ v g ->
    let dot = sum (g * v) in
    v * (g - dot))
;;

let softmax_axis x ~axis =
  let open Ndarray in
  let open Infix in
  Node.unary x ~op:"softmax_axis" ~f:(softmax_axis ~axis) ~df:(fun _ v g ->
    let dot = sum_axis (g * v) ~keepdim:true ~axis in
    v * (g - dot))
;;

module Infix = struct
  let ( + ) = add
  let ( - ) = sub
  let ( * ) = mul
  let ( / ) = div
  let ( ^ ) = powf
  let ( ~- ) = neg
  let ( @ ) = matmul
end

let backward = Autograd.backward

module Debug = struct
  let to_dot root =
    let nodes =
      Autograd.topo_sort root |> List.mapi ~f:(fun i x -> x, i)
    in
    let search x =
      List.find_map_exn nodes ~f:(fun (x', i) ->
        if phys_equal x x' then Some i else None)
    in
    let shape_label x =
      Ndarray.shape x.value
      |> Array.to_list
      |> List.map ~f:Int.to_string
      |> String.concat ~sep:" "
    in
    let node_line (x, i) =
      let label =
        Printf.sprintf
          "%s\nshape=(%s)\nrequires_grad=%b\ngrad=%b"
          x.op
          (shape_label x)
          x.requires_grad
          (Option.is_some x.grad)
        |> String.escaped
      in
      Printf.sprintf "  n%d [label=\"%s\"];" i label
    in
    let edge_lines (x, i) =
      List.map x.parents ~f:(fun parent ->
        Printf.sprintf "  n%d -> n%d;" (search parent) i)
    in
    let node_lines = List.map nodes ~f:node_line in
    let edge_lines = List.concat_map nodes ~f:edge_lines in
    String.concat
      ~sep:"\n"
      ([ "digraph autograd {"; "  rankdir=LR;" ]
       @ node_lines
       @ edge_lines
       @ [ "}" ])
  ;;
end
