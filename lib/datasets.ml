open! Base

module N = Ndarray

let make_moons ~n ~noise =
  if n % 2 <> 0 then invalid_arg "Datasets.make_moons: n must be even";
  let half = n / 2 in
  let theta = N.linspace ~start:0.0 ~stop:Float.pi ~num:half in
  let points x y = N.stack [| x; y |] ~axis:1 in
  let upper = points (N.map theta ~f:Float.cos) (N.map theta ~f:Float.sin) in
  let lower =
    points
      (N.map theta ~f:(fun t -> 1.0 -. Float.cos t))
      (N.map theta ~f:(fun t -> 0.5 -. Float.sin t))
  in
  let x =
    let open N.Infix in
    N.concat [| upper; lower |] ~axis:0 + (N.randn [| n; 2 |] * N.s noise)
  in
  let y = N.concat [| N.zeros [| half; 1 |]; N.ones [| half; 1 |] |] ~axis:0 in
  x, y
;;
