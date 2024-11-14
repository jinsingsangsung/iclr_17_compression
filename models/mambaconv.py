import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from functools import partial

from einops import rearrange, repeat
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.modules.conv import _ConvNd
from typing import Optional, List, Tuple, Union
from mamba_ssm.modules.mamba_simple import Mamba, Block
# from pos_embedding import build_position_encoding
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from math import prod
from math import pi, sqrt, ceil
# from mamba_ssm.utils.generation import GenerationMixin
# from mamba_ssm.models.mixer_seq_simple import _init_weights
# from mamba_ssm.utils import InferenceParams

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class MambaConv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        rms_norm: bool = True,
        dim_preserve: bool = False,
        drop_path: float = 0.,
        exag: bool = False,
        d_state=None,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
        self.dim_preserve = dim_preserve
        # assert out_channels % in_channels == 0, \
        #     f"output channel size {out_channels} must be divisible by input channel size {in_channels}"
        if d_state is None:
            d_state = kernel_size**2 // 2
        num_kernels = 1
        if exag:
            kernel_size = kernel_size * 4
            stride = stride * 4
            self.kernel_size = (kernel_size, kernel_size)
            self.stride = (stride, stride)
        # self.mamba_kernels = nn.ModuleList([
        #     partial(Mamba, d_state=d_state, layer_idx=layer_idx, **factory_kwargs)(in_channels)
        #     for layer_idx in range(num_kernels)
        # ])
        # self.pos_embed = build_position_encoding(N_steps=in_channels)
        # self.pos_linear = nn.Conv2d(in_channels*2, in_channels, 1, 1)
        # self.agg_token_pe = nn.Parameter(torch.zeros(1, 1, in_channels))  
        self.agg_token = nn.Parameter(torch.zeros(1, 1, in_channels))
        self.pos_embed = nn.Parameter(torch.zeros(1, kernel_size**2, 4*in_channels))
        trunc_normal_(self.pos_embed, std=.02)      
        
        mamba_channels = 4*in_channels
        self.mamba_kernels = nn.ModuleList([
            create_block(
                d_model=mamba_channels,
                d_state=d_state,
                layer_idx = i,
                rms_norm = rms_norm,
                bimamba_type = "v2",
                if_devide_out=True,
                **factory_kwargs,
            )
            for i in range(num_kernels)
        ])
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            mamba_channels, eps=1e-5, **factory_kwargs
        )
        self.linear = nn.Conv2d(mamba_channels, out_channels, 1, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if dim_preserve:
            self.conv = nn.Conv2d(4*in_channels, out_channels, 1, 1)
        else:
            if exag:
                kernel_size = kernel_size // 4
            self.conv = nn.Conv2d(4*in_channels, out_channels, kernel_size, kernel_size)

    def fused_add_norm(self, hidden_states, residual):
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.drop_path(hidden_states)
        return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))   
        # return residual

    def forward(self, input: Tensor) -> Tensor:
        '''
        input: torch.Tensor of size [B, C, H, W]
        '''
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        else:
            input = F.pad(input, self.padding*2)
        
        # patchify input sequences
        input = input.unfold(dimension=2, size=self.kernel_size[0], step=self.stride[0])
        input = input.unfold(dimension=3, size=self.kernel_size[1], step=self.stride[1])
        B, C, H2, W2, K1, K2 = input.shape
        
        # pos_embedding = self.pos_linear(self.pos_embed(input))
        pos_embedding = self.pos_embed.expand(B*H2*W2, -1, -1)
        # pos_embedding = repeat(pos_embedding, "B C K1 K2 -> B C H2 W2 K1 K2", H2=H2, W2=W2)
        # pos_embedding = rearrange(pos_embedding, "B C H2 W2 K1 K2 -> (B H2 W2) (K1 K2) C")
        # agg_token_pe = self.agg_token_pe.expand(B*H2*W2, -1, -1)
        agg_token = self.agg_token.expand(B*H2*W2, -1, -1)
        # input = rearrange(input, "B C H2 W2 K1 K2 -> (B H2 W2) (K1 K2) C")
        input = rearrange(input, "B C H2 W2 K1 K2 -> (B H2 W2) K1 K2 C")
        ssm_output = []
        for i, dir in enumerate([(),(1),(2),(1,2)]):
            dir_input = input.flip(dir)
            ssm_output.append(dir_input)
        input = torch.cat(ssm_output, dim=-1).flatten(1,2)
        
        # input = torch.cat((input, agg_token), dim=1)
        # pos = torch.cat((pos_embedding, agg_token_pe), dim=1)
        pos = pos_embedding
        # input = torch.cat((input, pos), dim=-1)
        input = input + pos
        output = torch.cat([rearrange(self.fused_add_norm(*kernel(input)), "(B H2 W2) (K1 K2) C -> B C (H2 K1) (W2 K2)", H2=H2, W2=W2, K1=K1, K2=K2)
                for kernel in self.mamba_kernels], dim=1)        
        output = self.conv(output)
        # output = self.linear(output)
        return output
    
    def dim_preserving_forward(self, input: Tensor) -> Tensor:
        '''
        input: torch.Tensor of size [B, C, H, W]
        '''
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        else:
            input = F.pad(input, self.padding*2)

        # patchify input sequences
        H, W = input.shape[-2:]
        input = input.unfold(dimension=2, size=self.kernel_size[0], step=self.stride[0])
        input = input.unfold(dimension=3, size=self.kernel_size[1], step=self.stride[1])
        B, C, H2, W2, K1, K2 = input.shape
        # pos_embedding = self.pos_linear(self.pos_embed(input))
        pos_embedding = self.pos_embed.expand(B*H2*W2, -1, -1)
        # pos_embedding = repeat(pos_embedding, "B C K1 K2 -> B C H2 W2 K1 K2", H2=H2, W2=W2)
        # pos_embedding = rearrange(pos_embedding, "B C H2 W2 K1 K2 -> (B H2 W2) (K1 K2) C")
        input = rearrange(input, "B C H2 W2 K1 K2 -> (B H2 W2) (K1 K2) C")
        
        # pos = torch.cat((pos_embedding[:, :agg_token_position, :], agg_token_pe, pos_embedding[:, agg_token_position:, :]), dim=1)
        pos = pos_embedding
        try:
            input = input + pos
        except:
            import pdb; pdb.set_trace()
        output = torch.cat([rearrange(self.fused_add_norm(*kernel(input)), "(B H2 W2) (K1 K2) C -> B C (H2 K1) (W2 K2)", H2=H2, W2=W2, K1=K1, K2=K2)
                for kernel in self.mamba_kernels], dim=1)
        # output = F.fold(output, (H, W), (K1, K2), stride=(K1, K2))
        return output    

class MambaGlobalConv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: tuple,
        kernel_size: _size_2_t = 1,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        rms_norm: bool = True,
        drop_path: float = 0.,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
        # assert out_channels % in_channels == 0, \
        #     f"output channel size {out_channels} must be divisible by input channel size {in_channels}"
        kernel_size = prod(spatial_dim)
        d_state = min(kernel_size // 2, 256)
        num_kernels = 1
        # self.mamba_kernels = nn.ModuleList([
        #     partial(Mamba, d_state=d_state, layer_idx=layer_idx, **factory_kwargs)(in_channels)
        #     for layer_idx in range(num_kernels)
        # ])
        # self.pos_embed = build_position_encoding(N_steps=in_channels)
        self.pos_embed = nn.Parameter(torch.zeros(1, kernel_size, in_channels))
        # self.pos_linear = nn.Conv2d(in_channels*2, in_channels, 1, 1)
        # self.agg_token_pe = nn.Parameter(torch.zeros(1, 1, in_channels))
        trunc_normal_(self.pos_embed, std=.02)        

        self.agg_token = nn.Parameter(torch.zeros(1, 1, in_channels))
        self.mamba_kernels = nn.ModuleList([
            create_block(
                d_model=in_channels,
                d_state=d_state,
                layer_idx = i,
                rms_norm = rms_norm,
                bimamba_type = "v2",
                if_devide_out=True,
                **factory_kwargs,
            )
            for i in range(num_kernels)
        ])
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            in_channels, eps=1e-5, **factory_kwargs
        )        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.token_position = "middle"
        # self.token_position = "end"
        # self.token_position = "first"

    def fused_add_norm(self, hidden_states, residual):
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.drop_path(hidden_states)
        return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))   

    def forward(self, input: Tensor) -> Tensor:
        '''
        input: torch.Tensor of size [B, C, H, W]
        '''
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        else:
            input = F.pad(input, self.padding*2)

        B, _, H2, W2 = input.shape
        
        # pos_embedding = self.pos_linear(self.pos_embed(input))
        pos_embedding = self.pos_embed.expand(B, -1, -1)
        # pos_embedding = repeat(pos_embedding, "B C K1 K2 -> B C H2 W2 K1 K2", H2=H2, W2=W2)
        # pos_embedding = rearrange(pos_embedding, "B C H2 W2 K1 K2 -> (B H2 W2) (K1 K2) C")
        # agg_token_pe = self.agg_token_pe.expand(B, -1, -1)
        # agg_token = self.agg_token.expand(B, -1, -1)
        input = rearrange(input, "B C H2 W2 -> B (H2 W2) C")
        # pos = torch.cat((pos_embedding[:, :agg_token_position, :], agg_token_pe, pos_embedding[:, agg_token_position:, :]), dim=1)
        pos = pos_embedding
        input = input + pos
        output = torch.cat([rearrange(self.fused_add_norm(*kernel(input)), "B (H2 W2) C -> B C H2 W2", H2=H2, W2=W2)
                for kernel in self.mamba_kernels], dim=1)       
        # output = self.conv(output)
        return output

class MambaUpConv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        rms_norm: bool = True,
        drop_path: float = 0.,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
        mamba_channels = in_channels
        d_state = kernel_size**2 // 2
        num_kernels = 1
        self.mamba_kernels = nn.ModuleList([
            create_block(
                d_model=mamba_channels,
                d_state=d_state,
                layer_idx = i,
                rms_norm = rms_norm,
                bimamba_type = "v2",
                if_devide_out=True,
                **factory_kwargs,
            )
            for i in range(num_kernels)
        ])
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            mamba_channels, eps=1e-5, **factory_kwargs
        )
        self.ke = nn.Embedding(kernel_size**2, in_channels, device=device, dtype=dtype).weight
        self.conv = nn.Conv2d(mamba_channels, out_channels, 1, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ps = nn.PixelShuffle(kernel_size)

    def fused_add_norm(self, hidden_states, residual):
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.drop_path(hidden_states)
        return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

    def forward(self, input):
        B, C, H, W = input.shape
        K1 = self.kernel_size[0]
        K2 = self.kernel_size[1]
        assert C == self.in_channels
        input = rearrange(input, "B C H W -> (B H W) C")[:, None]
        while input.size(1) < K1*K2:
            # import pdb; pdb.set_trace()
            input = torch.cat([input]+[self.fused_add_norm(*kernel(input))[:,-1,:][:, None] for kernel in self.mamba_kernels], dim=1)
        ke = repeat(self.ke, "N C -> BHW N C", BHW=B*H*W)
        # input = rearrange(torch.cat([input, ke], dim=1), "BHW (K1 K2) C -> BHW K1 K2 C", K1=K1, K2=K2)
        input = input+ke
        # ssm_output = []
        # for i, dir in enumerate([(),(1),(2),(1,2)]):
        # # for i, dir in enumerate([()]):
        #     dir_input = input.flip(dir)
        #     ssm_output.append(dir_input)
        # input = torch.cat(ssm_output, dim=-1).flatten(1,2)
        # output = torch.cat([rearrange(self.fused_add_norm(*kernel(input)), "(B H W) (K1 K2) C -> B C (K1 H) (K2 W)", B=B, H=H, W=W, K1=K1, K2=K2)
        #         for kernel in self.mamba_kernels], dim=1)
        # input = rearrange(input, "(B H W) (K1 K2) C -> B C (K1 H) (K2 W)", B=B, H=H, W=W, K1=K1, K2=K2)
        input = rearrange(input, "(B H W) (K1 K2) C -> B (C K1 K2) H W", B=B, H=H, W=W, K1=K1, K2=K2)
        input = self.ps(input)
        output = self.conv(input)
        return output


# class MambaKernel(nn.Module, GenerationMixin):
#     def __init__(
#         self,
#         d_model,
#         n_layer,
#         d_intermediate,
#         d_state,
#         rms_norm,
#         initializer_cfg=None,
#         device=None,
#         dtype=None,
#     ) -> None:
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         mamba_channels = 4*d_model
#         self.backbone = nn.ModuleList([
#             create_block(
#                 d_model=mamba_channels,
#                 d_state=d_state,
#                 layer_idx = i,
#                 rms_norm = rms_norm,
#                 bimamba_type = "v2",
#                 if_devide_out=True,
#                 **factory_kwargs,
#             )
#             for i in range(n_layer)
#         ])
#         self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
#             mamba_channels, eps=1e-5, **factory_kwargs
#         )        

#         self.apply(
#             partial(
#                 _init_weights,
#                 n_layer=n_layer,
#                 **(initializer_cfg if initializer_cfg is not None else {}),
#                 n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
#             )
#         )
#         self.tie_weights()

#     def tie_weights(self):
#         if self.config.tie_embeddings:
#             self.lm_head.weight = self.backbone.embedding.weight

#     def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
#         return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

#     def fused_add_norm(self, hidden_states, residual):
#         if residual is None:
#             residual = hidden_states
#         else:
#             residual = residual + self.drop_path(hidden_states)
#         return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))   

#     def forward(self, input, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
#         """
#         "position_ids" is just to be compatible with Transformer generation. We don't use it.
#         num_last_tokens: if > 0, only return the logits for the last n tokens
#         """
#         hidden_states = torch.cat([self.fused_add_norm(*kernel(input, inference_params=inference_params, **mixer_kwargs)) for kernel in self.backbone], dim=-1)
#         if num_last_tokens > 0:
#             hidden_states = hidden_states[:, -num_last_tokens:]
#         return hidden_states

# class MambaUpConv(_ConvNd):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  d_state: int,
#                  kernel_size: int,
#                  stride: int,
#                  rms_norm: bool,
#                  device=None,
#                  dtype=None,
#                  ):
#         super(MambaUpConv, self).__init__()
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         mamba_channels = 4*in_channels
#         num_kernels = 1
#         self.mamba_kernels = MambaKernel(
#                                 mamba_channels,
#                                 1,
#                                 0,
#                                 d_state,
#                                 rms_norm, 
#         )
#         self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
#             mamba_channels, eps=1e-6, **factory_kwargs
#         )

#     def forward(self, x):
#         """
#         input: B C H W
#         output: B C' Hk Wk
#         """
#         x = rearrange(x, "B C H W -> (B H W) C")[:, None]

#         return self.upconv(x)


def create_block(
    d_model,
    d_state,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    bimamba_type=None,
    if_devide_out=False,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, d_state=d_state, layer_idx=layer_idx, bimamba_type=bimamba_type, 
                        if_devide_out=if_devide_out, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

# def main():
#     B, C, H, W = 2, 3, 150, 320
#     input = torch.rand(B, C, H, W).cuda()
#     mamba_convolution = MambaConv2D(in_channels=3,
#                                     out_channels=12,
#                                     kernel_size=5,
#                                     stride=1,
#                                     padding=1,
#                                     device=input.device)
#     output = mamba_convolution(input)
    

# if __name__ == '__main__':
#     main()
        