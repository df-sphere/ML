# Transformer Notes

## 1. Main Input Parameters

Common transformer parameters:

- `V`: vocabulary size
- `T`: maximum sequence length / maximum context length
- `H`: hidden dimension, also called model dimension
- `N`: minibatch size
- `h`: number of attention heads
- `d_ff`: hidden size of the feed-forward network

Typical shapes:

- input token ids: `(N, T)`
- embeddings: `(N, T, H)`
- transformer layer input/output: `(N, T, H)`

The embedding layer uses:

```text
Embedding(V, H)
```

so the embedding table has shape:

```text
(V, H)
```

`T` is the maximum sequence length the model is designed to support. If you want the model to handle up to 1000 tokens, then typically `T = 1000`.

You can train with sequences shorter than `T`, but if you want strong performance at length `T`, then training should include sequences near that length. Otherwise the model may support that context length in theory, but not use it well in practice.

## 2. What Is a Head vs a Layer?

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

## 3. Single Attention Head

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

## 4. How H Maps to Q, K, V, and W_O

The hidden dimension `H` is the size of the token representation going into the attention block.

If:

```text
X shape = (N, T, H)
```

then the query, key, and value projections are usually:

```text
W_Q shape = (H, H)
W_K shape = (H, H)
W_V shape = (H, H)
```

and:

```text
Q = XW_Q
K = XW_K
V = XW_V
```

So `Q`, `K`, and `V` also have hidden size `H` before being split across heads.

If the model has `h` heads, then usually:

```text
head_dim = H / h
```

Each head works on one slice of size `head_dim`.

After all heads are concatenated back together, the concatenated tensor has size `H` again, and the output projection is usually:

```text
W_O shape = (H, H)
```

So:

- `H` determines the input size to `Q`, `K`, and `V`
- `H` is split across the heads
- `W_O` maps the concatenated multi-head output back into size `H`

## 5. Multi-Head Attention

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

## 6. Transformer Block

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

## 7. Multi-Head Attention vs Feed-Forward Network

These are two different parts of the transformer layer.

A self-attention block by itself is **not** a full transformer layer. A cross-attention block by itself is also **not** a full transformer layer.

A full transformer layer includes:

- attention sublayer
- residual connection
- layer normalization
- feed-forward network
- another residual connection
- another layer normalization

In an encoder, the attention sublayer is self-attention.

In a decoder, the layer usually contains:

- masked self-attention
- cross-attention to the encoder output
- feed-forward network

### Multi-Head Attention

Multi-head attention is **not**:

```text
Linear(H, d_ff) -> activation -> Linear(d_ff, H)
```

Instead, multi-head attention does:

```text
Q = XW_Q
K = XW_K
V = XW_V
head_i = softmax(Q_i K_i^T / sqrt(d_k)) V_i
MultiHead(X) = concat(head_1, ..., head_h) W_O
```

So multi-head attention uses:

- the query, key, and value projections
- attention scores and softmax
- concatenation of all heads
- the output projection `W_O`

### Feed-Forward Network

The feed-forward network is the part that uses two linear layers:

```text
FFN(X) = Linear(H, d_ff) -> activation -> Linear(d_ff, H)
```

So `d_ff` is an input parameter of the transformer layer, and it belongs only to the feed-forward network, not to the multi-head attention formula.

## 8. Regular LLM vs Machine Translation Transformer

This distinction is one of the most important uses of transformers.

### A. Regular LLM Architecture

A regular decoder-only LLM has:

- token embeddings
- stacked decoder blocks
- output projection to vocabulary logits

It does **not** have a separate encoder for a source sentence.

The model receives one sequence and predicts the next token of that same sequence. Examples:

- text completion
- chat response generation
- code generation

If the current hidden representation inside the decoder is `X'`, then self-attention uses:

```text
Q = X' W_Q
K = X' W_K
V = X' W_V
```

The key point is:

```text
K = X' W_K
```

So the LLM attends only to the current token stream it is generating from. All queries, keys, and values come from the same running representation.

During inference, the model generates one token at a time. After it predicts the next token, that predicted token is appended to the sequence and fed back into the model as part of the next input.

Example:

```text
Input:  "The capital of France is"
Output: "Paris"
Next input: "The capital of France is Paris"
```

Then the model predicts the next token after `"Paris"`.

### B. Machine Translation Architecture

A machine translation transformer is usually an **encoder-decoder** model.

It has:

- an encoder that reads the source sentence
- a decoder that generates the target sentence

Example:

- source language: French
- target language: English

The encoder reads the full source sentence first, for example:

```text
Je mange une pomme
```

and produces encoder outputs:

```text
E
```

Then the decoder generates the translated sentence token by token:

```text
I
I eat
I eat an
I eat an apple
```

During inference, the decoder also works autoregressively. After it predicts one target token, that token is fed back into the decoder input for the next step.

Example:

```text
Decoder input: <BOS>
Predicts: I

Decoder input: <BOS> I
Predicts: eat

Decoder input: <BOS> I eat
Predicts: an

Decoder input: <BOS> I eat an
Predicts: apple
```

So both systems reuse their own generated output during inference, but the translation decoder also has access to the separate encoder output `E`.

Inside the decoder, there are two attention types:

- masked self-attention over the partial target sentence
- cross-attention to the encoder output

In that cross-attention:

```text
Q = X' W_Q
K = E W_K
V = E W_V
```

The key point is:

```text
K = E W_K
```

So the decoder is not attending only to its own generated text. It is explicitly attending to the encoder's representation of the source sentence.

### C. What the Encoder Does

The encoder:

- reads the input sentence
- builds contextual representations of the source tokens
- produces the encoder output `E`

For translation, `E` is the memory of the source sentence that the decoder can consult.

### D. What the Decoder Does

The decoder:

- sees the previously generated target tokens
- uses masked self-attention so it cannot look at future target tokens
- uses cross-attention to look at `E`
- predicts the next target token

So in translation, the decoder does two jobs at once:

- keep track of the target sentence built so far
- look back at the source sentence through the encoder output

During inference, each newly predicted target token is appended to the decoder input and used to predict the next target token.

### E. Main Formula Difference to Remember

For a regular decoder-only LLM:

```text
Q = X' W_Q
K = X' W_K
V = X' W_V
```

So:

```text
K = X' W_K
```

For a translation model decoder doing cross-attention:

```text
Q = X' W_Q
K = E W_K
V = E W_V
```

So:

```text
K = E W_K
```

This is the architectural difference:

- regular LLM: one stream, no separate encoder memory
- translation model: decoder attends to a separate encoder memory

### F. Simple Side-by-Side Example

#### Regular LLM

Prompt:

```text
The capital of France is
```

The decoder-only LLM uses its current hidden states `X'` from that same prompt history:

```text
Q = X' W_Q
K = X' W_K
V = X' W_V
```

and predicts:

```text
Paris
```

It is only continuing the same sequence.

At inference time, the predicted token is fed back as part of the next input sequence.

#### Translation Model

Source sentence:

```text
Je mange une pomme
```

Encoder output:

```text
E = Encoder(source sentence)
```

Decoder generates:

```text
I
I eat
I eat an
I eat an apple
```

When producing `"apple"`, the decoder forms its query from its current target-side hidden state:

```text
Q = X' W_Q
```

but the keys come from the encoded source sentence:

```text
K = E W_K
```

So the decoder can attend strongly to the source token representation corresponding to `"pomme"`.

At inference time, each predicted English token is fed back into the decoder input, while the encoder output `E` stays fixed as the representation of the French source sentence.

## 9. Stacked Self-Attention and Cross-Attention

When transformer layers are stacked, the output of one layer becomes the input to the next layer.

For stacked self-attention layers:

```text
Q = X' W_Q
K = X' W_K
V = X' W_V
```

where `X'` is the output from the previous transformer layer.

For cross-attention in the decoder:

```text
Q = X' W_Q
K = E W_K
V = E W_V
```

where:

- `X'` is the current decoder representation
- `E` is the encoder output

So:

- in self-attention, `Q`, `K`, and `V` all come from the same current representation
- in cross-attention, `Q` comes from the decoder, while `K` and `V` come from the encoder

If there are multiple decoder layers, the decoder representation keeps changing from one layer to the next. That updated decoder representation is used to form the new `Q` in the next decoder layer, while the encoder output continues to provide `K` and `V`.

## 10. Important Correction

The residual connection is usually added **after combining all heads**, not separately inside each head.

So instead of:

```text
x11 = x11 + z
```

for each head, the standard transformer does:

```text
X' = X + MultiHead(X)
```

## 11. Compact Summary

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

## 12. Intuition

- `Q` asks what each token is looking for.
- `K` tells what each token offers.
- `V` is the information passed along.
- Softmax turns scores into attention weights.
- Multi-head attention lets the model focus on different relationships at the same time.
- Residual connections help preserve information across layers.
- Layer normalization stabilizes training.
