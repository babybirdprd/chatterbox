# Adapted from https://raw.githubusercontent.com/xingchensong/S3Tokenizer/v0.1.7/s3tokenizer/model.py
# Original Apache License applies.

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
# from einops import rearrange # Not used in the copied S3TokenizerV2Base or its direct dependencies
from torch import Tensor, nn

# Changed from 'from .utils import make_non_pad_mask, mask_to_bias, onnx2torch'
from .s3_local_utils import make_non_pad_mask, mask_to_bias
# onnx2torch is not imported as init_from_onnx will be commented out.


@dataclass
class ModelConfig:
    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 6
    n_codebook_size: int = 4096

    use_sdpa: bool = False


class LayerNorm(nn.LayerNorm):

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):

    def _conv_forward(self, x: Tensor, weight: Tensor,
                      bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment *
                               torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[
        np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):

    def __init__(self, n_state: int, n_head: int, use_sdpa: bool = False):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

        self.use_sdpa = use_sdpa

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self,
                      q: Tensor,
                      k: Tensor,
                      v: Tensor,
                      mask: Optional[Tensor] = None):
        _, _, D_model = q.shape # Renamed D to D_model to avoid clash if used in other scopes
        scale = (D_model // self.n_head)**-0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k_orig_shape = k.shape # For SDPA path
        k = k.view(*k.shape[:2], self.n_head, -1) 
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if not self.use_sdpa:
            k = k.permute(0, 2, 3, 1) * scale # (B, n_head, d_head, T_k)
            qk = q @ k  # (B, n_head, T_q, T_k)
            if mask is not None:
                qk = qk + mask # mask is (B, 1, T_q, T_k) or (B, n_head, T_q, T_k)
            
            w = torch.nn.functional.softmax(qk.float(), dim=-1).to(q.dtype)
            return (w @ v).permute(0, 2, 1,
                                   3).flatten(start_dim=2), qk.detach()
        else:
            # For SDPA, k needs to be (B, n_head, T_k, d_head)
            k = k.permute(0, 2, 1, 3) * scale # (B, n_head, T_k, d_head)
            # q is (B, n_head, T_q, d_head)
            # v is (B, n_head, T_v, d_head) where T_k == T_v
            # mask for SDPA should be (B, n_head, T_q, T_k)
            
            # The original code had an assert mask is not None.
            # If mask is (B, 1, T_q, T_k) it needs broadcasting to (B, n_head, T_q, T_k)
            # If mask comes from mask_to_bias, it's already (B, 1, T_q_len, T_k_len) or similar
            # For self-attention T_q = T_k = T_v
            
            output = torch.nn.functional.scaled_dot_product_attention(
                q, # (B, n_head, T_q, d_head)
                k, # (B, n_head, T_k, d_head)
                v, # (B, n_head, T_v, d_head)
                attn_mask=mask, # Needs to be (B, n_head, T_q, T_k) or broadcastable
                dropout_p=0.,
                # scale=1., # scale is already applied to q and k
            )
            # output is (B, n_head, T_q, d_head)
            return output.permute(0, 2, 1, 3).flatten(start_dim=2), None # (B, T_q, n_head*d_head)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, n_state: int, n_head: int, use_sdpa: bool):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head, use_sdpa=use_sdpa)
        self.attn_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(),
                                 Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):

    def __init__(
        self,
        n_mels: int,
        n_ctx: int, # Max sequence length for positional embedding
        n_state: int,
        n_head: int,
        n_layer: int,
        stride: int, # Used in conv1
        use_sdpa: bool,
    ):
        super().__init__()
        self.stride = stride
        self.conv1 = Conv1d(n_mels,
                            n_state,
                            kernel_size=3,
                            stride=stride, # Will change sequence length
                            padding=1)
        self.conv2 = Conv1d(n_state,
                            n_state,
                            kernel_size=3,
                            stride=2, # Will change sequence length
                            padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList([
            ResidualAttentionBlock(n_state, n_head, use_sdpa=use_sdpa)
            for _ in range(n_layer)
        ])
        self.ln_post = LayerNorm(n_state) # Added from typical encoder structure, may not be in original ONNX

    def forward(self, x: Tensor, x_len: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x : torch.Tensor, shape = (batch_size, n_mels, T_mel)
        x_len: torch.Tensor, shape = (batch_size,) original lengths of T_mel
        """
        # Calculate masks based on original lengths
        # Mask for conv1 (based on original x_len)
        conv1_mask_bool = make_non_pad_mask(x_len, max_len=x.size(2)).unsqueeze(1) # (B, 1, T_mel)
        x = x * conv1_mask_bool.to(x.dtype)
        x = F.gelu(self.conv1(x))
        
        # Update lengths after conv1
        # L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        # kernel=3, padding=1, dilation=1 => L_out = floor((L_in - 1)/stride + 1)
        x_len_after_conv1 = (x_len - 1) // self.stride + 1

        # Mask for conv2 (based on x_len_after_conv1)
        conv2_mask_bool = make_non_pad_mask(x_len_after_conv1, max_len=x.size(2)).unsqueeze(1) # (B, 1, T_conv1_out)
        x = x * conv2_mask_bool.to(x.dtype)
        x = F.gelu(self.conv2(x))
        
        # Update lengths after conv2
        x_len_after_conv2 = (x_len_after_conv1 - 1) // 2 + 1
        
        x = x.permute(0, 2, 1)  # (B, T_encoded, n_state)

        # Create attention mask based on final lengths
        # This mask is (B, T_encoded)
        attn_mask_bool = make_non_pad_mask(x_len_after_conv2, max_len=x.shape[1]) 
        # mask_to_bias expects (B, T) or (B, 1, T, T) etc.
        # For self-attention, it should be (B, 1, T_encoded, T_encoded) for MHA or (B, T_encoded, T_encoded) for SDPA
        # The MultiHeadAttention in this code expects mask of shape (B, 1, T, T) for adding, 
        # or (B, n_head, T, T) for SDPA.
        # Let's assume mask_to_bias handles the expansion or MHA does.
        # The original whisper code had mask.unsqueeze(1).unsqueeze(2) to make it (B,1,1,T) then broadcast.
        # Or more simply (B, T_q, T_k) -> (B, 1, T_q, T_k) for MHA's additive bias.
        # For SDPA, it needs to be (B, N_HEADS, T_q, T_k) or broadcastable.
        # Given x_len_after_conv2 is (B,), make_non_pad_mask makes (B, T_encoded)
        # So, this needs to be expanded for attention.
        # (B, T_encoded) -> (B, 1, T_encoded, T_encoded) for additive bias
        # (B, T_encoded) -> (B, n_heads, T_encoded, T_encoded) for SDPA mask
        # The mask_to_bias in this code creates a bias for additive masking.
        # For self-attention, if mask is (B, T), it implies padding along T.
        # The mask for MHA should be (B, 1, T, T)
        # Original code in model.py: mask = mask_to_bias(mask, x.dtype) where mask was (B, T_final). This is not right for attention.
        # Corrected mask for self-attention:
        final_attn_mask_bool = attn_mask_bool[:, None, :] * attn_mask_bool[:, :, None] # (B, T_enc, T_enc)
        final_attn_mask_for_bias = final_attn_mask_bool.unsqueeze(1) # (B, 1, T_enc, T_enc)
        attention_bias = mask_to_bias(final_attn_mask_for_bias, x.dtype)


        # Apply positional embedding
        # Ensure positional_embedding is not longer than x
        current_seq_len = x.shape[1]
        pos_emb = self.positional_embedding[:current_seq_len, :]
        x = (x + pos_emb).to(x.dtype)

        for block in self.blocks:
            x = block(x, attention_bias) # Pass the bias mask

        x = self.ln_post(x) # Apply final LayerNorm
        
        # Mask out padding regions in the output sequence before returning
        output_mask_bool = make_non_pad_mask(x_len_after_conv2, max_len=x.size(1)).unsqueeze(2) # (B, T_encoded, 1)
        x = x * output_mask_bool.to(x.dtype)

        return x, x_len_after_conv2


class EuclideanCodebook(nn.Module):
    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        embed = torch.zeros(codebook_size, dim)
        self.codebook_size = codebook_size
        self.register_buffer("embed", embed)

    @torch.inference_mode()
    def preprocess(self, x: Tensor) -> Tensor:
        # Using einops.rearrange here: x = rearrange(x, "... d -> (...) d")
        # Manual rearrange:
        shape = x.shape
        d = shape[-1]
        return x.reshape(-1, d)


    @torch.inference_mode()
    def quantize(self, x: Tensor) -> Tensor:
        embed_t = self.embed.t() # (dim, codebook_size)
        dist = -(x.pow(2).sum(1, keepdim=True) - 2 * (x @ embed_t) +
                 embed_t.pow(2).sum(0, keepdim=True))
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    @torch.inference_mode()
    def postprocess_emb(self, embed_ind, original_shape):
        # original_shape is shape of x before preprocess: (..., d)
        # embed_ind is flat: ((...))
        # target_shape is (...), i.e. original_shape[:-1]
        target_shape = original_shape[:-1]
        return embed_ind.view(target_shape)

    @torch.inference_mode()
    def dequantize(self, embed_ind: Tensor) -> Tensor:
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    @torch.inference_mode()
    def encode(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        x_processed = self.preprocess(x)
        embed_ind = self.quantize(x_processed)
        embed_ind_reshaped = self.postprocess_emb(embed_ind, original_shape)
        return embed_ind_reshaped

    @torch.inference_mode()
    def decode(self, embed_ind: Tensor) -> Tensor:
        quantize = self.dequantize(embed_ind)
        return quantize


class VectorQuantization(nn.Module):
    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        self._codebook = EuclideanCodebook(dim=dim,
                                           codebook_size=codebook_size)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    @torch.inference_mode()
    def encode(self, x: Tensor) -> Tensor: # x is (B, T, dim)
        x = F.normalize(x, p=2, dim=-1)
        embed_in = self._codebook.encode(x) # embed_in is (B, T)
        return embed_in

    @torch.inference_mode()
    def decode(self, embed_ind: Tensor) -> Tensor: # embed_ind is (B, T)
        quantize = self._codebook.decode(embed_ind) # quantize is (B, T, dim)
        # Original code: quantize = rearrange(quantize, "b n d -> b d n")
        # This means (B, T, dim) -> (B, dim, T)
        quantize = quantize.permute(0, 2, 1)
        return quantize


# Renamed from S3Tokenizer to S3TokenizerV2Base
class S3TokenizerV2Base(nn.Module):
    def __init__(self, name: str, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.encoder = AudioEncoder(
            self.config.n_mels,
            self.config.n_audio_ctx,
            self.config.n_audio_state,
            self.config.n_audio_head,
            self.config.n_audio_layer,
            2 if name == "speech_tokenizer_v1_25hz" else 1, # Stride for conv1
            self.config.use_sdpa,
        )
        self.quantizer = VectorQuantization(self.config.n_audio_state,
                                            self.config.n_codebook_size)

    def forward(self, mel: Tensor, mel_len: Tensor) -> Tuple[Tensor, Tensor]:
        return self.quantize(mel, mel_len)

    @torch.inference_mode()
    def quantize(self, mel: Tensor, mel_len: Tensor) -> Tuple[Tensor, Tensor]:
        # mel: (B, n_mels, T_mel), mel_len: (B,)
        hidden, code_len = self.encoder(mel, mel_len) # hidden: (B, T_encoded, n_audio_state), code_len: (B,)
        code = self.quantizer.encode(hidden) # code: (B, T_encoded)
        return code, code_len

    @property
    def device(self):
        return next(self.parameters()).device

    # def init_from_onnx(self, onnx_path: str):
    #     # ckpt = onnx2torch(onnx_path, None, False) # Commented out due to onnx2torch dependency
    #     # self.load_state_dict(ckpt, strict=True)
    #     pass

    # def init_from_pt(self, ckpt_path: str):
    #     # ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
    #     # self.load_state_dict(ckpt, strict=True)
    #     pass

    def freeze(self):
        for _, param in self.named_parameters():
            param.requires_grad = False
