# Backprop Derivatives: Detailed Exam Guide

This guide explains how to decide whether to use pre-activation or post-activation in derivatives, and how to do backprop correctly for `sigmoid`, `tanh`, and `ReLU`.

## 1) Notation You Should Keep Consistent

For layer `l`:

- `Z[l] = A[l-1] @ W[l] + b[l]` (pre-activation, also called logits for output layer)
- `A[l] = g[l](Z[l])` (post-activation / layer output)
- `A[0] = X`

Loss for binary classification (mean BCE):

- `L = -(1/N) * sum(y*log(p) + (1-y)*log(1-p))`, where `p = A[L]`

Shape convention (batch-first):

- `A[l-1]`: `(N, d_in)`
- `W[l]`: `(d_in, d_out)`
- `b[l]`: `(d_out,)`
- `Z[l], A[l]`: `(N, d_out)`

## 2) Core Chain Rule Pattern

Always move gradients in this order:

1. Upstream gradient arrives as `dL/dA[l]`
2. Convert through activation: `dL/dZ[l] = dL/dA[l] * g'[l](Z[l])`
3. Convert through linear map:
   - `dW[l] = A[l-1]^T @ dZ[l]`
   - `db[l] = sum(dZ[l], dim=0)`
   - `dA[l-1] = dZ[l] @ W[l]^T`

The key point: the local derivative is mathematically with respect to the activation input `Z[l]`, even if you rewrite it using `A[l]`.

## 3) Pre-Activation vs Post-Activation: The Rule

Use this decision rule:

- Differentiate activation as `dA/dZ = g'(Z)` (always true conceptually).
- If `g'(Z)` has an equivalent formula using `A`, you may compute it from `A`.

So:

- **Conceptual variable**: pre-activation `Z`
- **Implementation convenience**: sometimes `A` is enough

## 4) Activation-Specific Details

## Sigmoid

- Forward: `A = sigmoid(Z)`
- Local derivative: `dA/dZ = sigmoid(Z) * (1 - sigmoid(Z))`
- Equivalent cached form: `dA/dZ = A * (1 - A)`

If your helper is:

```python
def sigmoid_grad(s):
    return s * (1 - s)
```

then `s` must be post-activation (`A`), not logits (`Z`).

## Tanh

- Forward: `A = tanh(Z)`
- Local derivative: `dA/dZ = 1 - tanh(Z)^2`
- Equivalent cached form: `dA/dZ = 1 - A^2`

So `tanh_grad(A)` is valid if your function expects activated value.

## ReLU

- Forward: `A = max(0, Z)`
- Local derivative: `dA/dZ = 1 if Z > 0 else 0`
- At `Z = 0`, derivative is undefined; in practice set to `0` (standard choice).

For ReLU, mask by `Z > 0` (or equivalently `A > 0` except exactly at zero).

## 5) Output Layer Shortcut (Binary BCE + Sigmoid)

For output logit `Z` and probability `p = sigmoid(Z)`:

- BCE gives `dL/dp = -(y/p - (1-y)/(1-p)) / N`
- Sigmoid gives `dp/dZ = p(1-p)`
- Multiply and simplify:
  - `dL/dZ = (p - y) / N`

That means either approach is valid:

1. Full chain: `dL/dZ = (dL/dp) * (dp/dZ)`
2. Shortcut: `dL/dZ = (p - y) / N`

Both are mathematically identical if implemented correctly.

## 6) Hidden Layer Backprop Template

For hidden layer `l`:

- `dZ[l] = dA[l] * g'[l](Z[l])`
- `dW[l] = A[l-1]^T @ dZ[l]`
- `db[l] = sum(dZ[l], dim=0)`
- `dA[l-1] = dZ[l] @ W[l]^T`

This is the only template you need to memorize.

## 7) Common Mistakes (Very Exam-Relevant)

- Passing `Z` into a grad helper that expects `A` (or vice versa).
- Mixing up shape comments (`input_size` vs batch size `N`).
- Forgetting division by `N` when loss is a mean.
- Summing biases over the wrong axis (`dim=0` needed for batch-first).
- Using `dL/dp` as if it were `dL/dZ` at the output.
- Inconsistent notation (`A`, `H`, `P`) causing wrong chain links.

## 8) Fast Sanity Checks

Use these checks after each backward step:

- Shapes:
  - `dW` matches `W`
  - `db` matches `b`
  - propagated gradient matches previous activation shape
- Magnitudes:
  - no `nan`/`inf`
  - gradients not all zeros (unless expected from ReLU dead units)
- Learning behavior:
  - first few iterations should usually decrease training loss

## 9) Exam Mini-Checklist

When solving by hand:

1. Write forward equations (`Z`, then `A`) per layer.
2. Write loss derivative at output.
3. Convert output gradient to `dZ` at output.
4. Apply linear-layer formulas (`dW`, `db`, previous gradient).
5. Repeat activation + linear backward for each hidden layer.
6. Check every tensor shape.

If you keep this order, you almost never get lost.

