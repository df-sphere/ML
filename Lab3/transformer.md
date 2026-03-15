# Transformer Notes

## 1. What Is a Head vs a Layer?

A **head** is one attention computation:

```text
head_i = softmax(Q_i K_i^T / sqrt(d_k)) V_i
```

So a head is the part that computes attention weights with softmax and then uses those weights to combine the values `V`.

A **transformer layer** is much bigger than one head. A layer contains:

- multi-head attention
- concatenation of all heads
- output projection / fully connected layer
- residual connection
- layer normalization
- feed-forward network
- another residual connection
- another layer normalization

The output of one transformer layer becomes the input to the next transformer layer.

## 1. Single Attention Head

Given an input matrix `X`, one attention head computes:

```text
Q = XW_Q
K = XW_K
V = XW_V
```

Then compute the attention scores:

```text
S = QK^T / sqrt(d_k)
```

Apply softmax row-wise:

```text
A = softmax(S)
```

Then compute the head output:

```text
Z = AV = softmax(QK^T / sqrt(d_k)) V
```

## 2. Multi-Head Attention

Each head does the same computation independently:

```text
Z_1, Z_2, ..., Z_h
```

Each `Z_i` is one head output. We integrate all heads by concatenating them:

```text
Z_cat = [Z_1; Z_2; ...; Z_h]
```

Apply the output projection (fully connected layer):

```text
MultiHead(X) = Z_cat W_O
```

## 3. Transformer Block

After multi-head attention, apply the residual connection and normalization:

```text
X' = X + MultiHead(X)
X'' = LayerNorm(X')
```

Then apply the feed-forward network:

```text
F = FFN(X'')
Y = X'' + F
out = LayerNorm(Y)
```

Here, `out` is the final output of this transformer layer, and it becomes the input to the next transformer layer.

## 4. Important Correction

The residual connection is usually added **after combining all heads**, not separately inside each head.

So instead of:

```text
x11 = x11 + z
```

for each head, the standard transformer does:

```text
X' = X + MultiHead(X)
```

## 5. Compact Summary

```text
head_i = softmax(Q_i K_i^T / sqrt(d_k)) V_i
multihead = concat(head_1, ..., head_h) W_O
X' = LayerNorm(X + multihead)
out = LayerNorm(X' + FFN(X'))
```

So the full flow is:

```text
input to layer l -> transformer layer l -> out -> input to layer l+1
```

## 6. Intuition

- `Q` asks what each token is looking for.
- `K` tells what each token offers.
- `V` is the information passed along.
- Softmax turns scores into attention weights.
- Multi-head attention lets the model focus on different relationships at the same time.
- Residual connections help preserve information across layers.
- Layer normalization stabilizes training.
