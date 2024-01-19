"""Computes the flops needed for training/running transformer networks."""

import collections

# We checked this code with TensorFlow"s FLOPs counting, although we had to
# correct for this issue: https://github.com/tensorflow/tensorflow/issues/22071
# Assumptions going into the FLOPs counting
#   - An "operation" is a mathematical operation, not a machine instruction. So
#     an "exp" takes one opp like and add, even though in practice an exp
#     might be slower. This is not too bad an assumption because
#     matrix-multiplies dominate the compute for most models, so minor details
#     about activation functions don"t matter too much. Similarly, we count
#     matrix-multiplies as 2*m*n flops instead of m*n, as one might if
#     if considering fused multiply-add ops.
#   - Backward pass takes the same number of FLOPs as forward pass. No exactly
#     right (e.g., for softmax cross entropy loss the backward pass is faster).
#     Importantly, it really is the same for matrix-multiplies, which is most of
#     the compute anyway.
#   - We assume "dense" embedding lookups (i.e., multiplication by a one-hot
#     vector). On some hardware accelerators, these dense operations are
#     actually faster than sparse lookups.
# Please open a github issue if you spot a problem with this code!

# I am not sure if the below constants are 100% right, but they are only applied
# to O(hidden_size) activations, which is generally a lot less compute than the
# matrix-multiplies, which are O(hidden_size^2), so they don't affect the total
# number of FLOPs much.

# random number, >=, multiply activations by dropout mask, multiply activations
# by correction (1 / (1 - dropout_rate))
DROPOUT_FLOPS = 4

# compute mean activation (sum), computate variance of activation
# (square and sum), bias (add), scale (multiply)
LAYER_NORM_FLOPS = 5

# GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
ACTIVATION_FLOPS = 8

# max/substract (for stability), exp, sum, divide
SOFTMAX_FLOPS = 5


class TransformerHparams(object):
  """Computes the train/inference FLOPs for transformers."""

  def __init__(self, h, l, s=512, v=30522, e=None, i=None, heads=None,
      head_size=None, output_frac=0.15625, sparse_embed_lookup=False,
      decoder=False):
    self.h = h  # hidden size
    self.l = l  # number of layers
    self.s = s  # sequence length
    self.v = v  # vocab size
    self.e = h if e is None else e  # embedding size
    self.i = h * 4 if i is None else i  # intermediate size
    self.kqv = h if head_size is None else head_size * heads  # attn proj sizes
    self.heads = max(h // 64, 1) if heads is None else heads  # attention heads
    self.output_frac = output_frac  # percent of tokens using an output softmax
    self.sparse_embed_lookup = sparse_embed_lookup  # sparse embedding lookups
    self.decoder = decoder  # decoder has extra attn to encoder states

  def get_block_flops(self, seq_len=None):
    seq_len = self.s if seq_len is None else seq_len
    """Get the forward-pass FLOPs for a single transformer block."""
    attn_mul = 2 if self.decoder else 1
    block_flops = dict(
        kqv=3 * 2 * self.h * self.kqv * attn_mul,
        kqv_bias=3 * self.kqv * attn_mul,
        attention_scores=2 * self.kqv * seq_len * attn_mul,
        attn_softmax=SOFTMAX_FLOPS * seq_len * self.heads * attn_mul,
        attention_dropout=DROPOUT_FLOPS * seq_len * self.heads * attn_mul,
        attention_scale=seq_len * self.heads * attn_mul,
        attention_weighted_avg_values=2 * self.h * seq_len * attn_mul,
        attn_output=2 * self.h * self.h * attn_mul,
        attn_output_bias=self.h * attn_mul,
        attn_output_dropout=DROPOUT_FLOPS * self.h * attn_mul,
        attn_output_residual=self.h * attn_mul,
        attn_output_layer_norm=LAYER_NORM_FLOPS * attn_mul,
        intermediate=2 * self.h * self.i,
        intermediate_act=ACTIVATION_FLOPS * self.i,
        intermediate_bias=self.i,
        output=2 * self.h * self.i,
        output_bias=self.h,
        output_dropout=DROPOUT_FLOPS * self.h,
        output_residual=self.h,
        output_layer_norm=LAYER_NORM_FLOPS * self.h,
    )
    return sum(block_flops.values()) * seq_len

  def get_embedding_flops(self, output=False, seq_len=None):
    seq_len = self.s if seq_len is None else seq_len
    """Get the forward-pass FLOPs the transformer inputs or output softmax."""
    embedding_flops = {}
    if output or (not self.sparse_embed_lookup):
      embedding_flops["main_multiply"] = 2 * self.e * self.v
    # input embedding post-processing
    if not output:
      embedding_flops.update(dict(
          tok_type_and_position=2 * self.e * (seq_len + 2),
          add_tok_type_and_position=2 * self.e,
          emb_layer_norm=LAYER_NORM_FLOPS * self.e,
          emb_dropout=DROPOUT_FLOPS * self.e
      ))
    # projection layer if e != h
    if self.e != self.h or output:
      embedding_flops.update(dict(
          hidden_kernel=2 * self.h * self.e,
          hidden_bias=self.e if output else self.h
      ))
      # extra hidden layer and output softmax
      if output:
        embedding_flops.update(dict(
            hidden_activation=ACTIVATION_FLOPS * self.e,
            hidden_layernorm=LAYER_NORM_FLOPS * self.e,
            output_softmax=SOFTMAX_FLOPS * self.v,
            output_target_word=2 * self.v
        ))
        return self.output_frac * sum(embedding_flops.values()) * seq_len
    return sum(embedding_flops.values()) * seq_len

  def get_binary_classification_flops(self, seq_len=None):
    seq_len = self.s if seq_len is None else seq_len
    classification_flops = dict(
        hidden=2 * self.h * self.h,
        hidden_bias=self.h,
        hidden_act=ACTIVATION_FLOPS * self.h,
        logits=2 * self.h
    )
    return sum(classification_flops.values()) * seq_len

  def get_train_flops(self, batch_size, train_steps, seq_len=None, discriminator=False):
    """Get the FLOPs for pre-training the transformer."""
    # 2* for forward/backward pass
    return 2 * batch_size * train_steps * (
        (self.l * self.get_block_flops(seq_len=seq_len)) +
        self.get_embedding_flops(output=False, seq_len=seq_len) +
        (self.get_binary_classification_flops(seq_len=seq_len) if discriminator else
         self.get_embedding_flops(output=True, seq_len=seq_len))
    )

  def get_infer_flops(self):
    """Get the FLOPs for running inference with the transformer on a
    classification task."""
    return ((self.l * self.get_block_flops()) +
            self.get_embedding_flops(output=False) +
            self.get_binary_classification_flops())

class FusedTransformerHparams(TransformerHparams):

  def __init__(self, layer_small, layer_hidden, h, l, s=512, v=30522, e=None, i=None, heads=None,
      head_size=None, output_frac=0.15625, sparse_embed_lookup=False,
      decoder=False):

      super().__init__(h=h, l=l, s=s, v=v, e=e, i=i, heads=heads,
        head_size=head_size, output_frac=output_frac, sparse_embed_lookup=sparse_embed_lookup,
        decoder=decoder)
      self.ls = layer_small
      self.hs = layer_hidden

  def get_block_flops(self, seq_len=None):
    seq_len = self.s if seq_len is None else seq_len
    """Get the forward-pass FLOPs for a single transformer block."""
    attn_mul = 2 if self.decoder else 1
    block_flops = dict(
        kqv=3 * 2 * self.h * self.kqv * attn_mul,
        kqv_bias=3 * self.kqv * attn_mul,
        attention_scores=2 * self.kqv * seq_len * attn_mul,
        attn_softmax=SOFTMAX_FLOPS * seq_len * self.heads * attn_mul,
        attention_dropout=DROPOUT_FLOPS * seq_len * self.heads * attn_mul,
        attention_scale=seq_len * self.heads * attn_mul,
        attention_weighted_avg_values=2 * self.h * seq_len * attn_mul,
        attn_output=2 * self.h * self.h * attn_mul,
        attn_output_bias=self.h * attn_mul,
        attn_output_dropout=DROPOUT_FLOPS * self.h * attn_mul,
        attn_output_residual=self.h * attn_mul,
        attn_output_layer_norm=LAYER_NORM_FLOPS * attn_mul,
        intermediate=2 * self.h * self.i,
        intermediate_act=ACTIVATION_FLOPS * self.i,
        intermediate_bias=self.i,
        output=2 * self.h * self.i,
        output_bias=self.h,
        output_dropout=DROPOUT_FLOPS * self.h,
        output_residual=self.h,
        output_layer_norm=LAYER_NORM_FLOPS * self.h
    )
    fuse_flops = dict(
      kqv=self.h * self.kqv * self.ls * 2 + (self.h-self.hs) * self.hs * self.hs * 2 * 2,
      kqv_bias=self.kqv * self.ls + (self.h-self.hs) * self.hs,
      attn_output=self.h * self.h * self.ls * 2 + (self.h-self.hs) * self.hs * self.hs * 2 * 2,
      attn_output_bias=self.h * self.ls + (self.h-self.hs) * self.hs,
      attn_output_layer_norm=self.h * self.ls * 2 + (self.h-self.hs) * self.hs * 2,
      intermediate=self.h * self.i * self.ls * 2 + (self.h-self.hs)* self.hs * self.i * 2 * 2,
      intermediate_bias=self.i * self.ls + (self.h-self.hs) * self.i,
      output=self.h * self.i * self.ls * 2 + (self.h-self.hs) * self.i * self.h * 2,
      output_bias=self.h * self.ls * 2 + (self.h-self.hs) * self.h,
      output_layer_norm=self.h * self.ls * 2 + (self.h-self.hs) * self.h
    )
    return sum(block_flops.values()) * seq_len + sum(fuse_flops.values())

def get_flops_computer(config):
    return TransformerHparams(
        config.hidden_size,  # hidden size
        config.num_hidden_layers,  # layers
        s=config.max_position_embeddings, # len of sequence
        v=config.vocab_size,  # vocab size
        i=config.intermediate_size,  # ff intermediate hidden size
        heads=config.num_attention_heads,  # heads/head size
    )

def get_train_flops_by_config(config, batch_size, num_steps, seq_len=128):
    return TransformerHparams(
        config.hidden_size,  # hidden size
        config.num_hidden_layers,  # layers
        s=seq_len, # len of sequence
        v=config.vocab_size,  # vocab size
        i=config.intermediate_size,  # ff intermediate hidden size
        heads=config.num_attention_heads,  # heads/head size
    ).get_train_flops(batch_size, num_steps)  # 1M steps with batch size 2048

def get_infer_flops_by_config(config, seq_len=128):
    return TransformerHparams(
        config.hidden_size,  # hidden size
        config.num_hidden_layers,  # layers
        s=seq_len, # len of sequence
        v=config.vocab_size,  # vocab size
        i=config.intermediate_size,  # ff intermediate hidden size
        heads=config.num_attention_heads,  # heads/head size
    ).get_infer_flops()  # 1M steps with batch size 2048
