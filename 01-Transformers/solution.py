# %% [markdown]
"""
# Exercise 8
## Part 1: Introduction to Transformers

In this part, we will build the core components of a transformer from scratch, understand their fundamental properties, and apply them to a simple sequence task.

In particular, we will:
- Implement scaled dot-product attention and understand how it works.
- Build a self-attention layer and show that it is permutation equivariant.
- Implement positional embeddings so to inject order information and break permutation equivariance.
- Assemble a full transformer encoder block from these components.
- Train a small transformer to classify whether sequences of numbers are sorted.


<div class="alert alert-danger">
Set your python kernel to <code>tracking</code>
</div>

Places where you are expected to write code are marked with
```
# TASK:
...
# END OF TASK
```

This notebook was originally written by Albert Dominguez Mantes for the 2026 edition of the course with many inputs from Anna Foix-Romero, Eduardo Hirata-Miyasaki, Jordão Bragantini, Teun Huijben and Federico Carrara.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

plt.rcParams["figure.figsize"] = (6, 4)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# %% [markdown]
"""
### 1.1) Introduction

In the previous exercises, we used convolutional neural networks (CNNs) and multi-layer perceptrons (MLPs) to process data. Both of these architectures have **fixed receptive fields**: a convolutional layer looks at a local neighborhood of fixed size, and a fully connected layer has a fixed number of inputs. This means they cannot dynamically decide which parts of the input are most relevant for computing each output.

_Attention_ is a mechanism that solves this limitation. Given a set of inputs, attention allows the model to learn which inputs are relevant for each output. In other words, every element in the input can directly influence every output element, weighted by learned relevance scores.

The key idea is simple: attention computes a weighted sum of input values, where the weights are determined by the similarity between a query and a set of keys. One way to think about it: the query asks "what am I looking for?", the keys say "what do I contain?", and the values say "what do I return?".

This will be important and needed in Part 2 for tracking: given cell detections across time frames, we want the model to decide which detections in one frame are relevant to which detections in another frame.

Before diving into transformers, let's define an important term: _token_. A token is the minimal unit on which transformers perform computations. For text inputs (e.g. LLMs), an input token is a sequence of characters. For images, it's a small patch containing multiple pixels (usually 64 or 256). Each input token is then transformed to a vector of fixed dimension (sometimes called "token embedding"). As we will see, what transformers do is to iteratively transform this vector representation of a token. Note that for simplicity _token_ can be also used to refer to _token embeddings_ (hence the distinction _input token_ before).
"""

# %% [markdown]
"""
### 1.2) Scaled Dot-Product Attention

The most widely used form of attention is scaled dot-product attention (SDPA), introduced in the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017). Given matrices of queries $Q$, keys $K$, and values $V$, it computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where $d_k$ is the dimension of the keys. Each row of these matrices corresponds to one token, so all matrices have the same number of rows $N$. The scaling factor $\sqrt{d_k}$ basically prevents the dot products from growing too large in magnitude, which would push the softmax into regions with very small gradients.

<div>
    <img src="assets/attn_op.png" width="600"/>
</div>

Let's start by creating some toy data to work with: a batch of random tokens (token embeddings).
"""

# %%
torch.manual_seed(42) # for reproducibility

B = 2   # batch size
N = 6   # sequence length (number of tokens)
D = 8   # embedding dimension

# A batch of random token embeddings
X = torch.randn(B, N, D)
print(f"Input shape: {X.shape}")  # (B, N, D)

# %% [markdown]
"""
<div class="alert alert-block alert-info">
    <h2>Task 1.1</h2>

Implement Scaled Dot-Product Attention
</div>

Implement the function `scaled_dot_product_attention(Q, K, V)` that computes the attention output and the attention weights. The function should:

1. Compute the attention scores: $QK^\top / \sqrt{d_k}$, where $d_k$ is the feature (last) dimension of $K$.
2. Apply softmax along the last dimension to get the attention weights.
3. Multiply the weights by $V$ to get the output.
4. Return both the output and the attention weights.

All inputs have shape `(B, N, D)` where `B` is the batch size, `N` is the sequence length, and `D` is the embedding dimension.

Hint: $K^\top$ denotes transposition of the last two dimensions of $K$ (see `Tensor.transpose`'s documentation).
"""


# %% tags=["task"]
def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention.

    Args:
        Q: Query tensor, shape (B, N, D)
        K: Key tensor, shape (B, N, D)
        V: Value tensor, shape (B, N, D)

    Returns:
        Tuple of (output, attention_weights), both tensors.
        output has shape (B, N, D), attention_weights has shape (B, N, N).
    """
    # TASK: implement scaled dot-product attention
    attn_weights = None
    output = None
    # END OF TASK
    return output, attn_weights


# %% tags=["solution"]
def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention.

    Args:
        Q: Query tensor, shape (B, N, D)
        K: Key tensor, shape (B, N, D)
        V: Value tensor, shape (B, N, D)

    Returns:
        Tuple of (output, attention_weights), both tensors.
        output has shape (B, N, D), attention_weights has shape (B, N, N).
    """
    d_k = K.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (B, N, N)
    attn_weights = torch.softmax(scores, dim=-1)  # (B, N, N)
    output = torch.matmul(attn_weights, V)  # (B, N, D)
    return output, attn_weights


# %%
# Let's test your implementation by comparing it to PyTorch's built-in
# scaled_dot_product_attention (which does not return the weights, but
# we can compare the output).
Q, K, V = X, X, X  # for the SDPA sanity check we use Q=K=V; SelfAttention will project them separately

our_output, our_weights = scaled_dot_product_attention(Q, K, V)
torch_output = F.scaled_dot_product_attention(Q, K, V)

assert our_output.shape == (B, N, D), f"Output shape should be {(B, N, D)}, got {our_output.shape}"
assert our_weights.shape == (B, N, N), f"Weights shape should be {(B, N, N)}, got {our_weights.shape}"
assert torch.allclose(our_output, torch_output, atol=1e-5), "Output does not match PyTorch's implementation!"
print("Your attention implementation is correct!")

# %% [markdown]
"""
Let's visualize the attention weights for the first sample in the batch. Each row shows how much each token "attends to" every other token.
"""

# %%
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(our_weights[0].detach().numpy(), cmap="viridis", vmin=0)
ax.set_xlabel("Key position")
ax.set_ylabel("Query position")
ax.set_title("Attention weights (sample 0)")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
<div class="alert alert-block alert-warning">
    <b>Question:</b>
    The attention weights seem to all be between 0 and 1 in this case. Is this always true for any given query and key matrices?
</div>
"""

# %% [markdown] tags=["solution"]
"""
<div class="alert alert-block alert-warning">
    <b>Answer:</b>
    Yes, it is always true as they are the output of a softmax function, which normalizes the scores over the keys for each query.
"""

# %% [markdown]
"""
### 1.3) Self-Attention

In the example above, we used the same tensor `X` for queries, keys, and values. Using the raw input directly is obviously limiting, we'd ideally want the model to learn different representations for queries, keys, and values to enhance expressability.

An option to do this is to add simple learned linear projections: three separate weight matrices $W_Q$, $W_K$, $W_V$ which transform the input $X$ into queries, keys, and values respectively:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

<div>
    <img src="assets/matrix_form.png" width="900"/>
</div>

This is called _self-attention_: the attention mechanism acts on transformed versions ($Q$, $K$, $V$) of the same input $X$.

A variant of attention used mostly in decoder architecture is _cross-attention_, in which the query matrix $Q$ comes from one input $X$ while the keys and values matrices ($K$, $V$) come from another input $X'$. We will focus on self-attention for now though.

<div class="alert alert-block alert-info">
    <h2>Task 1.2</h2>

Implement a Self-Attention Layer
</div>

Implement a `SelfAttention` module with:
- Three `nn.Linear` layers (without bias) that project the input into $Q$, $K$, $V$.
- A `forward` method that applies the projections and then calls your `scaled_dot_product_attention` function from Task 1.1.
"""


# %% tags=["task"]
class SelfAttention(nn.Module):
    def __init__(self, d_model: int):
        """Initialize self-attention layer.

        Args:
            d_model: Dimension of the input embeddings.
        """
        super().__init__()
        # TASK: define three linear projections (Q, K, V), each mapping d_model -> d_model, without bias
        # END OF TASK

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention.

        Args:
            x: Input tensor, shape (B, N, D)

        Returns:
            Output tensor, shape (B, N, D)
        """
        # TASK: project x into Q, K, V, then apply scaled_dot_product_attention
        # Hint: scaled_dot_product_attention returns a tuple (output, weights), but we only need the output here
        output = None
        # END OF TASK
        return output


# %% tags=["solution"]
class SelfAttention(nn.Module):
    def __init__(self, d_model: int):
        """Initialize self-attention layer.

        Args:
            d_model: Dimension of the input embeddings.
        """
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention.

        Args:
            x: Input tensor, shape (B, N, D)

        Returns:
            Output tensor, shape (B, N, D)
        """
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        output, _ = scaled_dot_product_attention(Q, K, V)
        return output


# %%
self_attn = SelfAttention(d_model=D)
Y = self_attn(X)
assert Y.shape == X.shape, f"Output shape should be {X.shape}, got {Y.shape}"
print(f"Input shape:  {X.shape}")
print(f"Output shape: {Y.shape}")
print("Self-attention layer seems to be implemented correctly!")

# %% [markdown]
"""
### 1.4) Permutation Equivariance

Let's derive a fundamental property of self-attention. Revisiting the SDPA formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Since $Q$, $K$, and $V$ are all derived from the same input $X$ (via linear projections), what happens if we change the order of tokens in $X$ (so, shuffling rows of $X$)? Intuitively, the dot products $QK^\top$ should be the same (just shuffled), so the output will be shuffled in the same way.

More formally, if $\pi$ is a permutation and $X_\pi$ denotes $X$ with its tokens reordered according to $\pi$, then:

$$\text{SelfAttention}(X_\pi) = \text{SelfAttention}(X)_\pi$$

This property is called **permutation equivariance**: permuting the input permutes the output in the same way. In other words, self-attention treats its input as a set: there is no notion of order.

<div class="alert alert-block alert-info">
    <h2>Task 1.3</h2>

Empirically show permutation equivariance
</div>

Verify this empirically:
1. Compute `Y = self_attn(X)` for our test input.
2. Create a random permutation `perm` of the token indices `[0, 1, ..., N-1]`. _Hint_: use `torch.randperm()`.
3. Permute the input (`X_perm`) by indexing `X` with `perm`. Note that `X` is of shape (B,N,D).
4. Compute `Y_perm` as the self-attention of `X_perm`.
5. Check that the output is permuted in the same way as the input.
"""


# %% tags=["task"]
torch.manual_seed(123)

# TASK: demonstrate permutation equivariance
# 1. Compute Y from X
Y = None

# 2. Create a random permutation of token indices
perm = None

# 3. Permute the input tokens
X_perm = None

# 4. Compute Y_perm from the permuted input
Y_perm = None

# 5. Check that the output is permuted in the same way as the input.
# (Hint: use torch.allclose)

# END OF TASK


# %% tags=["solution"]
torch.manual_seed(123)

# 1. Compute Y from X
Y = self_attn(X)

# 2. Create a random permutation of token indices
perm = torch.randperm(N)
print(f"Permutation: {perm.tolist()}")

# 3. Permute the input tokens
X_perm = X[:, perm, :]

# 4. Compute Y_perm from the permuted input
Y_perm = self_attn(X_perm)

# 5. Check that the output is permuted in the same way as the input.
assert torch.allclose(Y_perm, Y[:, perm, :], atol=1e-5), (
    "Permutation equivariance does NOT hold, something is wrong!"
)
print("Permutation equivariance holds: permuting the input permutes the output identically.")


# %% [markdown]
"""
<div class="alert alert-block alert-warning">
    <b>Question:</b>
    If self-attention is permutation equivariant, what information about the input is it unable to capture? Why is this a problem if we want to model sequences (e.g., a sentence, a time series)?
</div>
"""

# %% [markdown] tags=["solution"]
"""
It is unable to capture the order of tokens in the input sequence. This is a problem when modeling sequences (e.g., sentences, time series) because the order of elements often carries significant semantic meaning. For example, in a sentence, changing the order of words can alter the meaning.
"""

# %% [markdown]
"""
<div class="alert alert-block alert-success">
<h2>Checkpoint 1</h2>

You have implemented SDPA from scratch, built a self-attention layer, and observed that self-attention is permutation equivariant, treating its input as a set with no notion of order.

This is actually a very powerful property when we want to process sets (which is what we'll partly use in the next part!). But for sequences where order matters, we need to fix this. That's what we'll do next.
</div>
"""

# %% [markdown]
"""
### 1.5) Positional Encoding

Since self-attention is permutation equivariant, it has no way of knowing the position of each token in the sequence. To inject this information, we add a _positional encoding_ to the input embeddings before feeding them to the SDPA operation.

The original transformer paper proposed sinusoidal positional encodings, defined as:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

where $pos$ is the position in the sequence and $i$ is the dimension index. Each position gets a unique encoding vector, and the sinusoidal pattern allows the model to learn to attend to relative positions. This positional encoding vector is then added to the input token embedding (i.e. $X'=X+PE(X)$), adding the notion of position to it.

There are also learned positional embeddings (a simple `nn.Embedding` layer indexed by position), which are used in many architectures. Another variant becoming the standard is Rotary Positional Embeddings (RoPE), which in a nutshell encodes relative position of tokens as angles between their embeddings.

Here we'll implement the sinusoidal version, which has the advantage of being parameter-free and generalizing to arbitrary sequence lengths.

<div class="alert alert-block alert-info">
    <h2>Task 1.4</h2>

Implement Sinusoidal Positional Encoding
</div>

Implement the function `sinusoidal_positional_encoding(N, D)` so that it returns a tensor of shape `(N, D)` containing the positional encodings.

Hint: the term $\frac{1}{10000^{2i/d_{model}}}$ can be computed more stably as $\exp\left(-\frac{2i}{d_{model}} \ln(10000)\right)$.
"""


# %% tags=["task"]
def sinusoidal_positional_encoding(
    N: int, D: int
) -> torch.Tensor:
    """Compute sinusoidal positional encodings.

    Args:
        N (int): Sequence length / number of tokens.
        D (int): Embedding dimension (must be even).

    Returns:
        torch.Tensor: Positional encoding tensor of shape (N, d_model).
    """
    assert D % 2 == 0, "d_model must be even for sinusoidal positional encoding"

    pe = torch.zeros(N, D)
    position = torch.arange(0, N).unsqueeze(1).float()  # (N, 1)

    # TASK: compute the positional encoding
    # 1. Compute the "div_term": exp(-i/D * 2*ln(10000)) for i = 0, 1, ..., D/2 - 1
    # 2. Fill even indices (pe[:, 0::2]) with sin(position * div_term)
    # 3. Fill odd indices (pe[:, 1::2]) with cos(position * div_term)
    # END OF TASK

    return pe


# %% tags=["solution"]
def sinusoidal_positional_encoding(
    N: int, D: int
) -> torch.Tensor:
    """Compute sinusoidal positional encodings.

    Args:
        N (int): Sequence length / number of tokens.
        D (int): Embedding dimension (must be even).

    Returns:
        torch.Tensor: Positional encoding tensor of shape (N, d_model).
    """
    assert D % 2 == 0, "d_model must be even for sinusoidal positional encoding"

    pe = torch.zeros(N, D)
    position = torch.arange(0, N).unsqueeze(1).float()  # (N, 1)

    div_term = torch.exp(
        torch.arange(0, D//2, 1).float() * (-2*np.log(10000.0) / D)
    )  # (D / 2,)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


# %%
pe = sinusoidal_positional_encoding(N=N, D=D)
assert pe.shape == (N, D), f"Shape should be ({N}, {D}), got {pe.shape}"

expected_pe = torch.Tensor(
       [[ 0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          1.0000e+00,  0.0000e+00,  1.0000e+00],
        [ 8.4147e-01,  5.4030e-01,  9.9833e-02,  9.9500e-01,  9.9998e-03,
          9.9995e-01,  1.0000e-03,  1.0000e+00],
        [ 9.0930e-01, -4.1615e-01,  1.9867e-01,  9.8007e-01,  1.9999e-02,
          9.9980e-01,  2.0000e-03,  1.0000e+00],
        [ 1.4112e-01, -9.8999e-01,  2.9552e-01,  9.5534e-01,  2.9995e-02,
          9.9955e-01,  3.0000e-03,  1.0000e+00],
        [-7.5680e-01, -6.5364e-01,  3.8942e-01,  9.2106e-01,  3.9989e-02,
          9.9920e-01,  4.0000e-03,  9.9999e-01],
        [-9.5892e-01,  2.8366e-01,  4.7943e-01,  8.7758e-01,  4.9979e-02,
          9.9875e-01,  5.0000e-03,  9.9999e-01]
])
assert torch.allclose(pe, expected_pe, atol=1e-5), "Positional encoding values are incorrect"
print("Positional encoding shape and values look correct!")

# %% [markdown]
"""
Let's visualize the positional encoding. Each row is a different absolute position, while columns denote different dimensions/features of the embeddings. You can see that lower dimensions oscillate at higher frequencies, while higher dimensions change more slowly. This gives the model both fine- and coarse-grained position information.
"""

# %%
# Visualize with a larger sequence for a clearer picture
pe_vis = sinusoidal_positional_encoding(N=64, D=32)

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(pe_vis.numpy(), cmap="RdBu", aspect="auto")
ax.set_xlabel("Embedding dimension")
ax.set_ylabel("Position")
ax.set_title("Sinusoidal positional encoding")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
<div class="alert alert-block alert-info">
    <h2>Task 1.5</h2>

Show that positional embeddings break permutation equivariance
</div>

Now let's repeat the permutation experiment from Task 1.3, but this time we add positional encodings (PEs) to the input before passing it through self-attention. Since the PEs are different for each position, permuting the tokens will change the position information they receive, and the output should no longer be simply a permuted version of the original.

1. Add the positional encoding to `X`: `X_pe = X + PE`
2. Compute self-attention on `X_pe` to compute `Y_pe`
3. Permute the tokens and add the positional encodings: `X_pe_perm = X_perm + PE` (note: permute the tokens, but not the positional encoding!)
4. Compute self-attention on `X_pe_perm` to compute `Y_pe_perm`
5. Check that `Y_pe_perm` is NOT equal to permuted `Y_pe`
"""


# %% tags=["task"]
pe = sinusoidal_positional_encoding(N=N, D=D)

# TASK: show that positional embeddings break permutation equivariance
# 1. Add positional encoding to X
X_pe = None

# 2. Compute Y_pe
Y_pe = None

# 3. Permute tokens but add same positional encoding
X_pe_perm = None

# 4. Compute Y_pe_perm
Y_pe_perm = None

# 5. Verify that equivariance no longer holds
# (Hint: torch.allclose)

# END OF TASK


# %% tags=["solution"]
pe = sinusoidal_positional_encoding(N=N, D=D)

# 1. Add positional encoding to X
X_pe = X + pe.unsqueeze(0)  # broadcast PE across batch

# 2. Compute Y_pe
Y_pe = self_attn(X_pe)

# 3. Permute tokens but add same positional encoding
X_pe_perm = X_perm + pe.unsqueeze(0)

# 4. Compute Y_pe_perm
Y_pe_perm = self_attn(X_pe_perm)

# 5. Verify that equivariance no longer holds
equivariant = torch.allclose(Y_pe_perm, Y_pe[:, perm, :], atol=1e-5)
assert not equivariant, (
    "Permutation equivariant still holds. Are you sure your positional encoding is correct?"
)
print("Permutation equivariance is broken: the model is now position-aware.")

# %% [markdown]
"""
<div class="alert alert-block alert-warning">
    <b>Question:</b>
    At this stage, self-attention with positional encodings can capture both content and position information. Is it translation equivariant, like a CNN? In other words, if we shift all tokens in the input by one position (e.g., from `[x1, x2, x3]` to `[x2, x3, x1]`), will the output be shifted in the same way? 
</div>
"""


# %% [markdown] tags=["solution"]
"""
<div class="alert alert-block alert-warning">
    <b>Answer:</b>
    No, it is not translation equivariant. Shifting all tokens by one position will change the positional encodings they receive, and thus the output will not simply be a shifted version of the original output.
</div>
"""

# %% [markdown]
"""
### 1.6) The Transformer Encoder Block

Now that we understand the core component of a transformer, self-attention, let's assemble them into a full **transformer encoder block**. The standard architecture is:

1. Layer Normalization on the input
2. Multi-Head Self-Attention (MHA)
3. Residual connection (add the input back to the output of MHA)
4. Layer Normalization on the result
5. Feed-Forward Network (FFN): two linear layers with a GELU activation in between
6. Residual connection (add step 3's output back)

<div>
    <img src="assets/transformer_block.png" width="900"/>
</div>

In equation form:
$$x_{\text{norm}} = \text{LN}(x)$$
$$x' = x + \text{MHA}(x_{\text{norm}}, x_{\text{norm}}, x_{\text{norm}})$$
$$\text{output} = x' + \text{FFN}(\text{LN}(x'))$$

_Multi-head_ attention is a simple extension of self-attention: instead of computing one set of $Q$, $K$, $V$ projections, we compute $h$ different sets (called "heads"), run attention independently on each, concatenate the results, and project back to the original dimension. With this setting, different heads may learn to attend to different types of relationships. Virtually all transformer-based models use multi-head attention, so in general you should always use it! 

We will use PyTorch's built-in `nn.MultiheadAttention` for this, since you already understand the "single-head" self-attention mechanism from the previous tasks (and you _really_ want to use the optimized self-attention kernels for a runtime boost!).

<div class="alert alert-block alert-info">
    <h2>Task 1.6</h2>

Build a Transformer Encoder Block
</div>

Implement the `TransformerBlock` module following the architecture described above. Use:
- [`nn.MultiheadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) for the multi-head attention. Note that this module expects input of shape `(N, B, D)` by default (sequence first), but do set `batch_first=True` when instantiating it to use `(B, N, D)`. Its `forward` method expects 3 inputs, which should all be the normalized version of X for self-attention. It outputs a tuple of two values, out of which you should only pick one.
- [`nn.LayerNorm`](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for layer normalization.
- Two `nn.Linear` layers with a `nn.GELU` activation for the feed-forward network. The hidden dimension of the FFN is typically 4x larger than the token embedding dimension.
"""


# %% tags=["task"]
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        """A single transformer encoder block.

        Args:
            d_model: Dimension of the input embeddings.
            num_heads: Number of attention heads.
            d_ff: Hidden dimension of the feed-forward network.
        """
        super().__init__()

        # TASK: define the layers of the transformer block
        # - Two LayerNorm layers
        self.ln1 = ... # First LayerNorm
        self.mha = ... # Multi-head attention layer (remember to set batch_first=True)
        self.ln2 = ... # Second LayerNorm
        self.ffn = nn.Sequential(
            ...
        ) # feed-forward network
        # END OF TASK

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transformer block.

        Args:
            x: Input tensor, shape (B, N, D)

        Returns:
            Output tensor, shape (B, N, D)
        """
        # TASK: implement the forward pass according to the equation above
        # Hint: when calling the forward pass of MHA, pass the same three
        # inputs, i.e. self.mha(x_norm, x_norm, x_norm)
        output = None
        # END OF TASK
        return output


# %% tags=["solution"]
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        """A single transformer encoder block.

        Args:
            d_model: Dimension of the input embeddings.
            num_heads: Number of attention heads.
            d_ff: Hidden dimension of the feed-forward network.
        """
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transformer block.

        Args:
            x: Input tensor, shape (B, N, D)

        Returns:
            Output tensor, shape (B, N, D)
        """
        x_norm = self.ln1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + attn_out

        x_norm = self.ln2(x)
        ff_out = self.ffn(x_norm)
        x = x + ff_out

        return x


# %%
block = TransformerBlock(d_model=D, num_heads=2, d_ff=32)
Y_block = block(X)
assert Y_block.shape == X.shape, f"Output shape should be {X.shape}, got {Y_block.shape}"
assert sum(p.numel() for p in block.parameters()) == 872, f"Unexpected number of parameters. Please re-check your implementation!"

print(f"Input shape:  {X.shape}")
print(f"Output shape: {Y_block.shape}")
print(f"Parameters:   {sum(p.numel() for p in block.parameters()):,}")
print("Transformer block seems to be implemented correctly!")

# %% [markdown]
"""
<div class="alert alert-block alert-success">
<h2>Checkpoint 2</h2>

You have built all the core components of a transformer encoder: scaled dot-product attention, self-attention with learned projections, sinusoidal positional encodings, and a full transformer block with multi-head attention, feed-forward layers, and residual connections.

Now let's put everything together and train a small transformer on a concrete task!
</div>
"""

# %% [markdown]
"""
### 1.7) Putting it together: is this sequence sorted?

To see the transformer in action, we'll train one on a very simple but illustrative classification task: classifying whether a sequence of integers is sorted in ascending order. Given a sequence like `[3, 15, 42, 63, 91]`, the model should predict "sorted" (label 1). Given `[42, 7, 91, 15, 63]`, it should predict "not sorted" (label 0).

As this is a sequence classification task, we will need to produce a single output from a set of token embeddings. We will achieve so by taking the average over the token dimension (`N`) at the last stage of the model after the transformer encoder blocks, followed by a shallow linear layer to predict the final class, similar to what one does for image classification tasks with CNNs.

<div class="alert alert-block alert-warning">
    <b>Question:</b>
    Take a moment to think about this setting. Are positional encodings necessary here to solve the task? Why/why not? Note that the model is not a "pure" transformer, but has extra operations (global average pooling + linear layer).
</div>
"""

# %%
N_TOKENS = 10     # length of each sequence
VOCAB_SIZE = 100  # number in the sequence will be in the interval [0, VOCAB_SIZE)


def generate_unsorted_sequences(num_samples: int, seq_len: int, vocab_size: int) -> torch.Tensor:
    """Generate unsorted sequences."""
    random_seq = torch.randint(0, vocab_size, (num_samples, seq_len))
    already_sorted_idxs = (random_seq[:, :-1] <= random_seq[:, 1:]).all(dim=-1)
    if already_sorted_idxs.any():
        random_seq[already_sorted_idxs] = random_seq[already_sorted_idxs][:, torch.randperm(seq_len)]
    return random_seq

def generate_sorted_detection_data(
    num_samples: int, seq_len: int, vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate data for the sorted binary classification task.

    Half of the samples are unsorted, half sorted. Labels: 1 = sorted, 0 = not sorted.

    Args:
        num_samples: Number of sequences to generate.
        seq_len: Length of each sequence.
        vocab_size: Range of integer values [0, vocab_size).

    Returns:
        inputs: Integer tensor of shape (num_samples, seq_len).
        targets: Long tensor of shape (num_samples,) with binary labels.
    """
    half = num_samples // 2

    # Random sequences, highly likely unsorted
    unsorted = generate_unsorted_sequences(half, seq_len, vocab_size)
    unsorted_labels = torch.zeros(half, dtype=torch.long)

    # Sorted sequences
    sorted_seqs = torch.randint(0, vocab_size, (num_samples - half, seq_len))
    sorted_seqs, _ = sorted_seqs.sort(dim=-1)
    sorted_labels = torch.ones(num_samples - half, dtype=torch.long)

    # Concatenate and shuffle
    inputs = torch.cat([unsorted, sorted_seqs], dim=0)
    targets = torch.cat([unsorted_labels, sorted_labels], dim=0)
    perm = torch.randperm(num_samples)
    return inputs[perm], targets[perm]


# Sanity check
sample_in, sample_tgt = generate_sorted_detection_data(6, N_TOKENS, VOCAB_SIZE)
for i in range(6):
    label = "sorted" if sample_tgt[i].item() == 1 else "not sorted"
    print(f"{sample_in[i].tolist()}  -> {label}")

# %% [markdown]
"""
<div class="alert alert-block alert-info">
    <h2>Task 1.7</h2>

Build a sequence classification transformer
</div>

Implement the `SequenceClassifier` model with the following components:

1. Turn each number in the input into a dense vector: use an `nn.Embedding` layer that maps integer inputs to dense vectors of size `d_model`.
2. Positional encoding: re-use your `sinusoidal_positional_encoding` function from Task 1.4. A Boolean flag `use_pe` will be used to optionally disable positional encodings.
3. Transformer blocks: A stack of `num_layers` `TransformerBlock` modules (from Task 1.6).
4. Classification head: A `nn.Linear` layer that maps from `d_model` to `num_classes` (2 in our case).

The forward pass should:
1. Embed the input integers: transform each integer into a dense vector.
2. Add the positional encoding (if `use_pe` is True).
3. Pass through all transformer blocks.
4. Average pool over the sequence dimension to get a single vector per sequence.
5. Apply a small classification head.

<div>
    <img src="assets/seq_cls.png" width="700"/>
</div>
"""


# %% tags=["task"]
class SequenceClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_tokens: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        num_classes: int = 2,
        use_pe: bool = True,
    ):
        """A small transformer for sequence classification.

        Args:
            vocab_size: Number of possible integer values.
            n_tokens: Length of input sequences.
            d_model: Embedding dimension.
            num_heads: Number of attention heads per block.
            num_layers: Number of transformer blocks.
            d_ff: Hidden dimension of the feed-forward network.
            num_classes: Number of output classes.
            use_pe: Whether to add positional encoding.
        """
        super().__init__()
        self.use_pe = use_pe
        if self.use_pe:
            pe = sinusoidal_positional_encoding(n_tokens, d_model)
            # the following tells Torch that this is not a learnable parameter, but it should be
            # saved with the model and moved to the appropriate device with .to() calls
            # note that this is accessible anywhere in the class as self.pe
            self.register_buffer("pe", pe)
            
        # TASK: define the other model components
        # a) Input token embedding: use an nn.Embedding() layer
        # b) Transformer blocks: a module list of num_layers TransformerBlock instances
        # c) Classification head: an nn.Linear layer mapping a d_model vector to num_classes 
        # END OF TASK

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Integer input tensor, shape (B, N)

        Returns:
            Logits tensor, shape (B, num_classes).
        """
        # TASK: implement the forward pass
        # 1. Embed the tokens into a dense vector
        # 2. Add positional encodings, if necessary
        # 3. Feed the tokens through each transformer block
        # 4. Average-pool over the sequence dimension
        # 5. Apply classification head
        output = None
        # END OF TASK
        return output


# %% tags=["solution"]
class SequenceClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_tokens: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        num_classes: int = 2,
        use_pe: bool = True,
    ):
        """A small transformer for sequence classification.

        Args:
            vocab_size: Number of possible integer values.
            n_tokens: Length of input sequences.
            d_model: Embedding dimension.
            num_heads: Number of attention heads per block.
            num_layers: Number of transformer blocks.
            d_ff: Hidden dimension of the feed-forward network.
            num_classes: Number of output classes.
            use_pe: Whether to add positional encoding.
        """
        super().__init__()
        self.use_pe = use_pe
        if self.use_pe:
            pe = sinusoidal_positional_encoding(n_tokens, d_model)
            # the following tells Torch that this is not a learnable parameter, but it should be
            # saved with the model and moved to the appropriate device with .to() calls
            # note that this is accessible anywhere in the class as self.pe, like a regular attribute
            self.register_buffer("pe", pe)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Integer input tensor, shape (B, n_tokens)

        Returns:
            Logits tensor, shape (B, num_classes).
        """
        h = self.token_emb(x)
        if self.use_pe:
            h = h + self.pe.unsqueeze(0)
        for block in self.blocks:
            h = block(h)
        h = h.mean(dim=1)  # average pool over the sequence dimension
        return self.head(h)


# %%
# Verify the model works
model = SequenceClassifier(
    vocab_size=VOCAB_SIZE,
    n_tokens=N_TOKENS,
    d_model=64,
    num_heads=4,
    num_layers=2,
    d_ff=128,
)
dummy_input = torch.randint(0, VOCAB_SIZE, (2, N_TOKENS))
dummy_output = model(dummy_input)
assert dummy_output.shape == (2, 2), (
    f"Output shape should be (2, 2), got {dummy_output.shape}"
)
assert sum(p.numel() for p in model.parameters()) == 73474, f"Unexpected number of parameters. Please re-check your implementation!"

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("SequenceClassifier works correctly!")
del model  # clean up the temporary model

# %% [markdown]
"""
Let's now train two models, one without and one with positional encodings. We provide the training loop for you.
"""

# %%
from torch.utils.data import TensorDataset, DataLoader
from IPython.display import clear_output, display

# Hyperparameters
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 128
BATCH_SIZE = 128
LR = 3e-4

# Generate data
train_inputs, train_targets = generate_sorted_detection_data(20000, N_TOKENS, VOCAB_SIZE)
val_inputs, val_targets = generate_sorted_detection_data(4000, N_TOKENS, VOCAB_SIZE)

train_ds = TensorDataset(train_inputs, train_targets) # constructs a torch Dataset object from input and target tensors
val_ds = TensorDataset(val_inputs, val_targets)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)


# %%
def compute_accuracy(model, dataloader, device):
    """Compute classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch).argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.numel()
    return correct / total


def train_model(
    model,
    train_dl,
    val_dl,
    num_epochs,
    lr,
    device,
    label,
    history,
    fig,
    ax1,
    ax2,
):
    """Train a model and live-update the shared loss/accuracy plot."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, val_accs = [], []
    history[label] = {"losses": train_losses, "accs": val_accs}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_dl))
        val_accs.append(compute_accuracy(model, val_dl, device))

        ax1.clear()
        ax2.clear()
        markers = {"Without PE": "s-", "With PE": "o-"}
        colors = {"Without PE": "tab:orange", "With PE": "tab:blue"}
        for run_label, run in history.items():
            ax1.plot(
                run["losses"],
                markers[run_label],
                color=colors[run_label],
                label=run_label,
            )
            ax2.plot(
                run["accs"],
                markers[run_label],
                color=colors[run_label],
                label=run_label,
            )

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training loss")
        ax1.legend()
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        ax2.axhline(
            y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random chance"
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Validation accuracy")
        ax2.set_ylim(0.4, 1.05)
        ax2.legend()
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        fig.tight_layout()
        clear_output(wait=True)
        display(fig)

    return train_losses, val_accs


# %%
NUM_EPOCHS = 15

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
history = {}

print("Training model without positional encoding...")
model_without_pe = SequenceClassifier(
    vocab_size=VOCAB_SIZE, n_tokens=N_TOKENS,
    d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, d_ff=D_FF,
    use_pe=False,
).to(device)
losses_without_pe, accs_without_pe = train_model(
    model_without_pe, train_dl, val_dl, NUM_EPOCHS, LR, device,
    label="Without PE", history=history, fig=fig, ax1=ax1, ax2=ax2,
)
print(f"  Final validation accuracy: {accs_without_pe[-1] * 100:.1f}%")

print("Training model with positional encoding...")
model_with_pe = SequenceClassifier(
    vocab_size=VOCAB_SIZE, n_tokens=N_TOKENS,
    d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, d_ff=D_FF,
    use_pe=True,
).to(device)
losses_with_pe, accs_with_pe = train_model(
    model_with_pe, train_dl, val_dl, NUM_EPOCHS, LR, device,
    label="With PE", history=history, fig=fig, ax1=ax1, ax2=ax2,
)
print(f"  Final validation accuracy: {accs_with_pe[-1] * 100:.1f}%")
plt.close(fig)

# %% [markdown]
"""
<div class="alert alert-block alert-warning">
    <b>Question:</b>
    Increase the number of training epochs to e.g. 50. What do you observe in the curves of the model without positional encoding? What is a possible explanation for this behaviour?
</div>
"""

# %% [markdown] tags=["solution"]
"""
<div class="alert alert-block alert-warning">
    <b>Answer:</b>
    The model without PEs shows the training loss converging to a small value similar to the one with PEs, but the validation accuracy remains around random chance (50%). The model is theoretically unable to discriminate between sorted and unsorted sequences, but it has enough capacity to simply memorize the training data sequences (i.e. it is overfitting). 
</div>
"""
# %% [markdown]
"""
<div class="alert alert-block alert-success">
<h2>Checkpoint 3</h2>

You have built a complete transformer encoder from scratch and trained it on a sequence classification task. By comparing models with and without positional encoding, you have seen that:

<ul>
    <li>Attention computes a learnable weighted sum where the weights represent relevance between tokens.</li>
    <li>Self-attention is permutation equivariant: it treats its input as a set with no notion of order.</li>
    <li>Positional embeddings break this equivariance, allowing the model to use position information when needed.</li>
    <li>A transformer block consists of a multi-head attention operation along with feed-forward layers, normalization, and a residual connection.</li>
</ul>

In Part 2, we will learn about tracking and use a transformer to solve the problem: instead of integers in a sequence, the tokens will correspond to cell detections in different frames, and the transformer will learn which cells across frames correspond to each other, which is similar to what <code>trackastra</code> does.
</div>
"""
