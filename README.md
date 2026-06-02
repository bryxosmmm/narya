# Narya

Narya is an experimental machine learning framework written in OCaml.

The project began as an attempt to understand the foundations of modern machine
learning systems by implementing their core components from first principles:
numerical arrays, tensors, automatic differentiation, and optimizers.

---

## Motivation

### Why OCaml?

The main reason is personal: I enjoy working with ML-family languages, and I
have enjoyed writing OCaml in the past. I also do not think Python is an ideal
language for the core of machine learning systems, even though it is the dominant
language in the ecosystem today.

This does not mean that OCaml is automatically the best language for this
domain. Narya is partly an exploration of what this kind of system feels like
when written in OCaml: with a strong type system, explicit data structures, and a
different set of trade-offs from the Python-based stack.

### Why not just use PyTorch?

I usually understand systems more deeply when I try to build a small version of
them myself.

Narya is not meant to replace PyTorch. It is a way to study the ideas behind
frameworks like PyTorch by implementing the core mechanisms directly. Reading
papers and documentation is useful, but implementing tensors, reverse-mode
automatic differentiation, and optimizers forces a different level of
understanding.

## Current status

Narya is in early development. I work on it in my spare time, mostly alongside
university, so the project is still experimental and changing quickly.

At the moment, the core pieces are beginning to work: there is a basic `Ndarray`
implementation, tensor operations, reverse-mode automatic differentiation, and a
small optimizer module. These pieces are already sufficient for simple training
examples, such as fitting a linear function with gradient descent.

The biggest weakness right now is ergonomics. The API still has rough edges,
some code requires too much boilerplate, and several abstractions will need to be
revisited as the project grows.

## Design
The design is still experimental. I am not trying to fix the full architecture
too early; instead, I am building the core pieces first and refining the
structure as the project grows.

The library is currently organized around three main components:

```text
Ndarray  - numerical storage, shapes, broadcasting, and primitive operations
Tensor   - computation graph, gradients, and reverse-mode autodiff
Optim    - optimizer state and parameter updates
```

The design is influenced by PyTorch, but the implementation is written directly
in OCaml and will likely diverge as the project develops.

## Examples

Narya currently includes two small examples:

- `train_linear` fits a linear function with gradient descent.
- `graph_debug` prints the autograd graph in DOT format.

Run them with:

```sh
dune exec train_linear
dune exec graph_debug
```

## Development

The current focus is to polish the existing core while gradually adding the
pieces needed for small neural networks: more optimizers, loss functions, and a
module abstraction for composing models.

I am also considering some PPX-based utilities where they can improve ergonomics
without hiding too much of the underlying system.
