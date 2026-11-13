# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Fuse Q/K RoPE at Megatron's existing attention boundary; keep cache generation."""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import attention
from megatron.core.transformer.attention import HAVE_FA3, apply_rotary_pos_emb, rearrange
from megatron.core.utils import deprecate_inference_params, is_fa_min_version, nvtx_range_pop, nvtx_range_push

from musa_titan.rl_align_kernels.kernels.rope.triton_rope import _positions_from_cu_seqlens, rope_forward
from musa_titan.utils.patching import git_commit, replace_attr

_PATCHED = False
_AUTHORED = {
    "author": "dinglei",
    "modified_date": "2026-11-13",
    "upstream": {"megatron": {
        "branch": "detached HEAD",
        "commit": "2fe64f17008845c7a71c97af75c2040087783510",
        "commit_date": "2026-07-21T17:09:34+08:00",
    }},
}


def _apply_paired_rope(module, query, key, rotary_pos_emb, packed_seq_params, inference_context):
    """Return a fused pair for supported self-attention, otherwise use native RoPE."""
    config = module.config
    if (
        inference_context is not None or config.flash_decode
        or config.rotary_interleaved or config.multi_latent_attention
        or getattr(config, "mrope_section", None) is not None
        or getattr(config, "fp8", None)
        or rotary_pos_emb is None or len(rotary_pos_emb) != 2
        or rotary_pos_emb[0] is not rotary_pos_emb[1]
        or not isinstance(rotary_pos_emb[0], Tensor)
        or query.device.type != "musa" or key.device != query.device
        or query.dtype not in (torch.bfloat16, torch.float32) or key.dtype != query.dtype
        or query.ndim not in (3, 4) or key.ndim != query.ndim
        or query.shape[:-2] != key.shape[:-2]
        or query.shape[-1] != 128 or key.shape[-1] != 128
        or query.numel() == 0 or key.numel() == 0
    ):
        return None
    cp = module.model_comm_pgs.cp
    if cp is not None and cp.size() != 1:
        return None
    if getattr(config, "context_parallel_size", 1) != 1:
        return None
    freqs = rotary_pos_emb[0]
    if (freqs.ndim != 4 or freqs.shape[1:] != (1, 1, 128)
            or freqs.device != query.device or freqs.requires_grad or freqs.shape[0] == 0):
        return None

    cu = None
    if packed_seq_params is not None:
        if packed_seq_params.qkv_format != "thd" or query.ndim != 3:
            return None
        cu = packed_seq_params.cu_seqlens_q_padded
        if cu is None:
            cu = packed_seq_params.cu_seqlens_q
        kv_cu = packed_seq_params.cu_seqlens_kv_padded
        if kv_cu is None:
            kv_cu = packed_seq_params.cu_seqlens_kv
        # Different Q/K segmentations require different positions; keep native calls.
        if (cu is None or cu is not kv_cu or cu.device != query.device
                or cu.ndim != 1 or cu.numel() < 2
                or cu.dtype not in (torch.int32, torch.int64)):
            return None
    elif query.ndim != 4 or freqs.shape[0] < query.shape[0]:
        return None

    # hack(dinglei)
    # Use Megatron's existing frequencies (including offsets/scaling). Only pack
    # the same dtype-rounded cos/sin into the paired kernel's required layout.
    # No inverse-frequency rebuild, CPU cache or persistent cache is introduced.
    angles = freqs[:, 0, 0, :64]
    cache = torch.cat((angles.cos().to(query.dtype), angles.sin().to(query.dtype)), dim=-1)
    if cu is None:
        positions = torch.arange(query.shape[0], device=query.device).repeat_interleave(query.shape[1])
    else:
        # Reset positions per packed (possibly padded) sequence, entirely on GPU.
        positions = _positions_from_cu_seqlens(cu, query.shape[0])
    q, k = rope_forward(
        positions.contiguous(), query.reshape(-1, query.shape[-2], 128).contiguous(),
        key.reshape(-1, key.shape[-2], 128).contiguous(), cache,
    )
    return q.reshape(query.shape), k.reshape(key.shape)
    #


def attention_forward(
    self,
    hidden_states: Tensor,
    attention_mask: Tensor,
    key_value_states: Optional[Tensor] = None,
    inference_context: Optional[BaseInferenceContext] = None,
    rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
    rotary_pos_cos: Optional[Tensor] = None,
    rotary_pos_sin: Optional[Tensor] = None,
    attention_bias: Optional[Tensor] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    sequence_len_offset: Optional[int] = None,
    *,
    inference_params: Optional[BaseInferenceContext] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Perform a forward pass through the attention module.

    Args:
        hidden_states (Tensor): Hidden states.
        attention_mask (Tensor): Attention mask.
        key_value_states (Optional[Tensor]): Key/value states (for cross attention).
        inference_context (Optional[BaseInferenceContext]): Inference context that manages
            KV cache.
        rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
            embedding tensor(s).
        rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
        rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
        attention_bias (Optional[Tensor]): Attention bias.
        packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
        sequence_len_offset (Optional[int]): Sequence length offset used for
            inference CUDA graphs.

    Return:
        (Tuple[Tensor, Tensor]) Attention output and bias.

    """
    # Check if we need to skip RoPE
    # no_rope is 0-indexed array and self.layer_number is 1-indexed
    no_rope = (
        self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
    )
    if no_rope:
        rotary_pos_emb = None

    inference_context = deprecate_inference_params(inference_context, inference_params)

    if inference_context and inference_context.is_dynamic_batching():
        assert HAVE_FA3 or is_fa_min_version(
            "2.7.3"
        ), "flash attn verion v2.7.3 and above is required for dynamic batching."

    # hidden_states: [sq, b, h]
    if self.config.flash_decode and not self.training and inference_context is not None:
        rotary_pos_emb = None
    else:
        assert rotary_pos_cos is None and rotary_pos_sin is None

    # For self attention we just duplicate the rotary_pos_emb if it isn't already
    if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
        rotary_pos_emb = (rotary_pos_emb,) * 2

    # =====================
    # Query, Key, and Value
    # =====================
    # Get the query, key and value tensors based on the type of attention -
    # self or cross attn.
    nvtx_range_push(suffix="qkv")
    query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)
    nvtx_range_pop(suffix="qkv")

    # ===================================================
    # Adjust key, value, and rotary_pos_emb for inference
    # ===================================================

    in_decode_mode = (
        inference_context is not None
        and inference_context.is_decode_only()
        and not self.training
    )

    # This branch only runs in the decode phase of flash decoding and returns after the linear
    # projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
    nvtx_range_push(suffix="adjust_key_value")
    if in_decode_mode and self.config.flash_decode:
        assert self.layer_number in inference_context.key_value_memory_dict
        assert inference_context.sequence_len_offset is not None
        inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[
            self.layer_number
        ]
        output = self.flash_decode(
            sequence_len_offset=sequence_len_offset,
            query_layer=query,
            key_layer=key,
            value_layer=value,
            inference_key_memory=inference_key_memory,
            inference_value_memory=inference_value_memory,
            rotary_cos=rotary_pos_cos,
            rotary_sin=rotary_pos_sin,
            rotary_interleaved=self.config.rotary_interleaved,
        )
        out = output.transpose(0, 1).contiguous()
        context_layer = out.view(out.size(0), out.size(1), -1)
        output, bias = self.linear_proj(context_layer)
        return output, bias

    if (
        in_decode_mode
        and self.config.enable_cuda_graph
        and self.config.cuda_graph_scope != "full_iteration"
        and inference_context.is_static_batching()
    ):
        raise ValueError(f"CUDA graphs must use flash decode with static batching!")

    query, key, value, rotary_pos_emb, attn_mask_type, block_table = (
        self._adjust_key_value_for_inference(
            inference_context,
            query,
            key,
            value,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )
    )

    if packed_seq_params is not None:
        query = query.squeeze(1)
        key = key.squeeze(1)
        value = value.squeeze(1)
    nvtx_range_pop(suffix="adjust_key_value")

    # ================================================
    # relative positional embedding (rotary embedding)
    # ================================================
    nvtx_range_push(suffix="rotary_pos_emb")
    # hack(dinglei)
    # Q and K are both available here after Megatron's inference/layout handling.
    # Fuse their rotations without changing QKV, attention, projection or caches.
    paired = _apply_paired_rope(
        self, query, key, rotary_pos_emb, packed_seq_params, inference_context,
    )
    if paired is not None:
        query, key = paired
        rotary_pos_emb = None  # Skip the two original rotations below.
    #
    if rotary_pos_emb is not None and not self.config.flash_decode:
        q_pos_emb, k_pos_emb = rotary_pos_emb

        if packed_seq_params is not None:
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        if q_pos_emb is not None:
            # TODO VIJAY: simplify
            if inference_context is None or inference_context.is_static_batching():
                query = apply_rotary_pos_emb(
                    query,
                    q_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_q,
                    cp_group=self.model_comm_pgs.cp,
                )
            else:
                query = inference_context.apply_rotary_emb_query(
                    query, q_pos_emb, self.config, cu_seqlens_q, self.model_comm_pgs.cp
                )
        if k_pos_emb is not None:
            key = apply_rotary_pos_emb(
                key,
                k_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
                cp_group=self.model_comm_pgs.cp,
            )

        # TODO, can apply positional embedding to value_layer so it has
        # absolute positional embedding.
        # otherwise, only relative positional embedding takes effect
        # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
    nvtx_range_pop(suffix="rotary_pos_emb")

    # ==================================
    # core attention computation
    # ==================================

    nvtx_range_push(suffix="core_attention")
    if self.checkpoint_core_attention and self.training:
        core_attn_out = self._checkpointed_attention_forward(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
    else:
        if inference_context is None or inference_context.is_static_batching():
            # Static batching attention kernel.
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        else:
            # Dynamic batching attention kernel.
            q, k, v = (query, key, value)
            cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
            cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

            core_attn_out = self.flash_decode_and_prefill(
                q,
                k,
                v,
                max_seqlen_q,
                max_seqlen_k,
                cu_query_lengths,
                cu_kv_lengths,
                kv_lengths,
                block_table,
            )
            core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

    if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
        # reshape to same output shape as unpacked case
        # (t, np, hn) -> (t, b=1, h=np*hn)
        # t is the pack size = sum (sq_i)
        # note that batch is a dummy dimension in the packed case
        core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
    nvtx_range_pop(suffix="core_attention")

    # =================
    # Output. [sq, b, h]
    # =================

    nvtx_range_push(suffix="linear_proj")
    output, bias = self.linear_proj(core_attn_out)
    if self.checkpoint_core_attention and self.training:
        self._discard_core_attention_output(output)
    nvtx_range_pop(suffix="linear_proj")

    return output, bias


def apply_patch():
    """Install before model construction through either public Titan entry point."""
    global _PATCHED
    if _PATCHED:
        return
    # hack(dinglei)
    # Patch the real Attention.forward inherited by SelfAttention so training
    # reaches the paired operator rather than merely exporting a standalone API.
    replace_attr(
        attention.Attention, "forward", attention_forward,
        reason="Fuse supported MUSA Q/K Neox RoPE with autograd; retain Megatron cache generation.",
        commits={"megatron": git_commit(attention.__file__)}, authored=_AUTHORED,
        upstream_path="megatron.core.transformer.attention.Attention.forward",
        current_path="musa_titan/megatron/rope.py", commit_branch_verified=True,
    )
    #
    _PATCHED = True
