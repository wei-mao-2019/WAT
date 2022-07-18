import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from models.mlp import MLP
from models.rnn import RNN
from utils.torch import *
import math
from torch import Tensor
from torch.nn import Parameter
import copy
from typing import Optional, Any


class CustomTransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(CustomTransformerEncoder, self).__init__()

        self.layers =  _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                cond: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask,
                         cond=cond)

        if self.norm is not None:
            output = self.norm(output)

        return output


class CustomTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 cond_dim=0, cond_type='cat'):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.cond_type = cond_type
        if cond_type == 'cat':
            self.linear1 = nn.Linear(d_model+cond_dim, dim_feedforward)
        elif cond_type =='add':
            # assert cond_dim == d_model
            self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # self.activation = _get_activation_fn(activation)
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CustomTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                cond: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if self.cond_type == 'cat':
            src2 = self.linear2(self.dropout(self.activation(self.linear1(torch.cat([src,cond],dim=-1)))))
        elif self.cond_type == 'add':
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src+cond))))

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class CustomTransformerEncoder_v2(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, cond_dim=None, d_model=None):
        super(CustomTransformerEncoder_v2, self).__init__()

        self.layers =  _get_clones(encoder_layer, num_layers)
        self.lin = _get_clones(nn.Linear(cond_dim,d_model,bias=True), num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                cond: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i, mod in enumerate(self.layers):
            cond_tmp = self.lin[i](cond)
            output = torch.cat([cond_tmp,output],dim=0)
            output = mod(output, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)[cond.shape[0]:]

        if self.norm is not None:
            output = self.norm(output)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Tranformer_VAE(nn.Module):
    """
    decode with action label
    v3_3 + batch norm
    act in posterior
    """
    def __init__(self, nx, ny, horizon, specs):
        super(Tranformer_VAE, self).__init__()
        self.nx = nx
        self.ny = ny
        # self.nz = nz
        self.horizon = horizon

        self.n_head = specs.get('n_head',4)
        self.n_layer = specs.get('n_layer',8)
        self.p_do = specs.get('p_do',0.1)
        self.nh_lin = specs.get('nh_lin',1024)
        self.nh_emb = specs.get('nh_emb',256)
        self.n_action = n_action = specs.get('n_action', 15)
        self.activation = specs.get('activation', 'gelu')

        enc_layer = nn.TransformerEncoderLayer(d_model=self.nh_emb,
                                               nhead=self.n_head,
                                               dim_feedforward=self.nh_lin,
                                               activation=self.activation)
        # history encoder
        self.his_enc_lin_proj = nn.Linear(self.nx, self.nh_emb)
        self.his_enc_transformer = nn.TransformerEncoder(encoder_layer=enc_layer,
                                                     num_layers=self.n_layer)

        # posterior
        self.mu_std_token = nn.Linear(self.n_action, self.nh_emb*2, bias=False)
        self.enc_lin_proj = nn.Linear(self.nx, self.nh_emb)
        self.enc_transformer = nn.TransformerEncoder(encoder_layer=enc_layer,
                                                     num_layers=self.n_layer)
        # decoder
        enc_layer = CustomTransformerEncoderLayer(d_model=self.nh_emb,
                                                  nhead=self.n_head,
                                                  dim_feedforward=self.nh_lin,
                                                  activation=self.activation,
                                                  cond_dim=self.nh_emb)
        self.dec_act_token = nn.Linear(self.n_action, self.nh_emb, bias=False)
        self.dec_lin_proj = nn.Linear(self.nh_emb, self.nx)
        self.dec_transformer = CustomTransformerEncoder(encoder_layer=enc_layer,
                                                        num_layers=self.n_layer)
        self.pe = PositionalEncoding(d_model=self.nh_emb)

    def encode_x(self, x):
        """
        x: seq_len*bs*dim
        """
        hx = self.his_enc_lin_proj(x)
        hx = self.pe(hx)
        hx = torch.mean(self.his_enc_transformer(hx),dim=0,keepdim=True)
        return hx

    def encode_y(self,hx, y, fn_mask, act):
        """
        y: seq_len*bs*dim
        fn_mask: seq_len*bs
        act: bs * naction
        """
        # hx = self.encode_x(x) # 1*bs*nh
        seq_len,bs,_ = y.shape
        hact = self.mu_std_token(act) #bs*2nh
        hact = hact.reshape([bs,-1,self.nh_emb]).transpose(0,1)
        hy = self.enc_lin_proj(y)
        hy = torch.cat([hact,hy],dim=0)
        hy = self.pe(hy)
        hy = hx+hy
        
        idx = [0]*2 + list(range(seq_len))
        fn_mask = fn_mask[:,idx]

        hmu = self.enc_transformer(hy.contiguous(),src_key_padding_mask=fn_mask.contiguous())[:2]
        mu = hmu[0]
        log_var = hmu[1]

        return mu, log_var

    def encode(self, x, y, act, fn_mask):
        hx = self.encode_x(x)
        emu,elogvar = self.encode_y(hx, y, fn_mask, act)
        return emu, elogvar, hx

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z, act, T, hx=None):
        """
        x: seq_len*bs*dim
        z: 1*bs*dim
        hx: 1*bs*dim
        """
        if hx is None:
            hx = self.encode_x(x)
        hact = self.dec_act_token(act)
        hact = hx+z+hact

        pe = self.pe.pe[:T]+hact
        hact = hact.repeat([T,1,1])
        hy = self.dec_transformer(pe, cond=hact)
        y = self.dec_lin_proj(hy)
        return y

    def forward(self, x, y, act, fn_mask):
        mu, logvar, hx = self.encode(x, y, act, fn_mask)
        z = self.reparameterize(mu, logvar) if self.training else mu
        yp = self.decode(x, z, act, y.shape[0], hx=hx)
        return yp,mu,logvar

    def sample_prior(self, x, act, T):
        """
        x: seq_len*bs*dim
        """
        _,bs,_ = x.shape
        z = torch.randn([1,bs,self.nh_emb])
        yp = self.decode(x, z, act, T)
        return yp


class Tranformer_VAE_v2(nn.Module):
    """
    decode with action label
    v3_3 + batch norm
    act in posterior
    """

    def __init__(self, nx, ny, horizon, specs):
        super(Tranformer_VAE_v2, self).__init__()
        self.nx = nx
        self.ny = ny
        # self.nz = nz
        self.horizon = horizon

        self.n_head = specs.get('n_head', 4)
        self.n_layer = specs.get('n_layer', 8)
        self.p_do = specs.get('p_do', 0.1)
        self.nh_lin = specs.get('nh_lin', 1024)
        self.nh_emb = specs.get('nh_emb', 256)
        self.n_action = n_action = specs.get('n_action', 15)
        self.activation = specs.get('activation', 'gelu')

        enc_layer = nn.TransformerEncoderLayer(d_model=self.nh_emb,
                                               nhead=self.n_head,
                                               dim_feedforward=self.nh_lin,
                                               activation=self.activation)

        # posterior
        self.mu_std_token = nn.Linear(self.n_action, self.nh_emb * 2, bias=False)
        self.enc_lin_proj = nn.Linear(self.nx, self.nh_emb)
        self.enc_transformer = nn.TransformerEncoder(encoder_layer=enc_layer,
                                                     num_layers=self.n_layer)
        # decoder
        enc_layer = CustomTransformerEncoderLayer(d_model=self.nh_emb,
                                                  nhead=self.n_head,
                                                  dim_feedforward=self.nh_lin,
                                                  activation=self.activation,
                                                  cond_dim=self.nh_emb,
                                                  cond_type='add')
        self.dec_act_token = nn.Linear(self.n_action, self.nh_emb, bias=False)
        self.dec_lin_proj_in = nn.Linear(self.nx,self.nh_emb)
        self.dec_lin_proj_out = nn.Linear(self.nh_emb,self.nx)
        self.dec_transformer = CustomTransformerEncoder(encoder_layer=enc_layer,
                                                        num_layers=self.n_layer)

        self.pe = PositionalEncoding(d_model=self.nh_emb)


    def encode(self, y, act,fn_mask):
        """
        y: seq_len*bs*dim (with history)
        fn_mask: seq_len*bs
        act: bs * naction
        """
        seq_len, bs, _ = y.shape
        hact = self.mu_std_token(act)  # bs*2nh
        hact = hact.reshape([bs, -1, self.nh_emb]).transpose(0, 1)
        hy = self.enc_lin_proj(y)
        hy = torch.cat([hact, hy], dim=0)
        hy = self.pe(hy)

        idx = [0] * 2 + list(range(seq_len))
        fn_mask = fn_mask[:, idx]

        hmu = self.enc_transformer(hy.contiguous(), src_key_padding_mask=fn_mask.contiguous())[:2]
        mu = hmu[0]
        log_var = hmu[1]

        return mu, log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z, act, T):
        """
        x: seq_len*bs*dim (only history)
        z: 1*bs*dim
        hx: 1*bs*dim
        """
        hact = self.dec_act_token(act)
        hact = z + hact # 1 * bs * nh
        hx = self.dec_lin_proj_in(x)
        idx = list(range(hx.shape[0]))
        idx = idx + idx[-1:]*T
        ht = hx[idx,:,:]
        # ht = torch.cat([hact[None,:,:],hx],dim=0)
        ht = self.pe(ht)
        hact = hact[None,:,:].repeat([ht.shape[0],1,1])
        att_mask = torch.tril()
        ht = self.dec_transformer(ht, cond=hact)[-T:]
        y = self.dec_lin_proj_out(ht)
        y = y + x[-1:]
        # y = []
        # for i in range(T):
        #     htt = self.pe(ht)
        #     hy = self.dec_transformer(htt)[-1:]
        #     ht = torch.cat([ht,hy.detach()],dim=0)
        #     yt = self.dec_lin_proj_out(hy)
        #     y.append(yt)
        #     print(i)
        # y = torch.cat(yt,dim=0)
        return y

    def forward(self, x, y, act, fn_mask):
        if fn_mask.shape[1] == y.shape[0]:
            # extend mask
            fn_mask = torch.cat([torch.zeros_like(x[:,:,0].transpose(0,1))==1,
                                 fn_mask],dim=1)

        mu, logvar = self.encode(torch.cat([x,y],dim=0), act, fn_mask)
        z = self.reparameterize(mu, logvar) if self.training else mu
        yp = self.decode(x, z, act, y.shape[0])
        return yp, mu, logvar

    def sample_prior(self, x, act, T):
        """
        x: seq_len*bs*dim
        """
        _, bs, _ = x.shape
        z = torch.randn([1, bs, self.nh_emb])
        yp = self.decode(x, z, act, T)
        return yp


class Tranformer_VAE_v3(nn.Module):
    """
    decode with action label
    v3_3 + batch norm
    act in posterior
    """

    def __init__(self, nx, ny, horizon, specs):
        super(Tranformer_VAE_v3, self).__init__()
        self.nx = nx
        self.ny = ny
        # self.nz = nz
        self.horizon = horizon

        self.n_head = specs.get('n_head', 4)
        self.n_layer = specs.get('n_layer', 8)
        self.p_do = specs.get('p_do', 0.1)
        self.nh_lin = specs.get('nh_lin', 1024)
        self.nh_emb = specs.get('nh_emb', 256)
        self.n_action = n_action = specs.get('n_action', 15)
        self.activation = specs.get('activation', 'gelu')

        enc_layer = nn.TransformerEncoderLayer(d_model=self.nh_emb,
                                               nhead=self.n_head,
                                               dim_feedforward=self.nh_lin,
                                               activation=self.activation)

        # posterior
        self.mu_std_token = nn.Linear(self.n_action, self.nh_emb * 2, bias=False)
        self.enc_lin_proj = nn.Linear(self.nx, self.nh_emb)
        self.enc_transformer = nn.TransformerEncoder(encoder_layer=enc_layer,
                                                     num_layers=self.n_layer)
        # decoder
        self.dec_act_token = nn.Linear(self.n_action, self.nh_emb, bias=False)
        self.dec_lin_proj_in = nn.Linear(self.nx,self.nh_emb)
        self.dec_lin_proj_out = nn.Linear(self.nh_emb,self.nx)
        self.dec_transformer = CustomTransformerEncoder_v2(encoder_layer=enc_layer,num_layers=self.n_layer,
                                                           cond_dim=self.nh_emb,d_model=self.nh_emb)

        self.pe = PositionalEncoding(d_model=self.nh_emb)


    def encode(self, y, act,fn_mask):
        """
        y: seq_len*bs*dim (with history)
        fn_mask: seq_len*bs
        act: bs * naction
        """
        seq_len, bs, _ = y.shape
        hact = self.mu_std_token(act)  # bs*2nh
        hact = hact.reshape([bs, -1, self.nh_emb]).transpose(0, 1)
        hy = self.enc_lin_proj(y)
        hy = torch.cat([hact, hy], dim=0)
        hy = self.pe(hy)

        idx = [0] * 2 + list(range(seq_len))
        fn_mask = fn_mask[:, idx]

        hmu = self.enc_transformer(hy.contiguous(), src_key_padding_mask=fn_mask.contiguous())[:2]
        mu = hmu[0]
        log_var = hmu[1]

        return mu, log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z, act, T):
        """
        x: seq_len*bs*dim (only history)
        z: 1*bs*dim
        hx: 1*bs*dim
        """
        hact = self.dec_act_token(act)
        hact = z + hact # 1 * bs * nh
        hx = self.dec_lin_proj_in(x)
        idx = list(range(hx.shape[0]))
        idx = idx + idx[-1:]*T
        ht = hx[idx,:,:]
        # ht = torch.cat([hact[None,:,:],hx],dim=0)
        ht = self.pe(ht)
        # hact = hact[None,:,:].repeat([ht.shape[0],1,1])
        mask_tmp1 = torch.zeros([x.shape[0]+1,ht.shape[0]+1],device=ht.device)
        mask_tmp1[:,:x.shape[0]+1] = 1
        mask_tmp2 = torch.tril(torch.ones([T,ht.shape[0]+1],device=ht.device),
                              diagonal=x.shape[0])
        attn_mask = torch.cat([mask_tmp1,mask_tmp2],dim=0)
        ht = self.dec_transformer(ht, mask=attn_mask==0, cond=hact[None,:,:])[-T:]
        y = self.dec_lin_proj_out(ht)
        y = y + x[-1:]
        # y = []
        # for i in range(T):
        #     htt = self.pe(ht)
        #     hy = self.dec_transformer(htt)[-1:]
        #     ht = torch.cat([ht,hy.detach()],dim=0)
        #     yt = self.dec_lin_proj_out(hy)
        #     y.append(yt)
        #     print(i)
        # y = torch.cat(yt,dim=0)
        return y

    def forward(self, x, y, act, fn_mask):
        if fn_mask.shape[1] == y.shape[0]:
            # extend mask
            fn_mask = torch.cat([torch.zeros_like(x[:,:,0].transpose(0,1))==1,
                                 fn_mask],dim=1)

        mu, logvar = self.encode(torch.cat([x,y],dim=0), act, fn_mask)
        z = self.reparameterize(mu, logvar) if self.training else mu
        yp = self.decode(x, z, act, y.shape[0])
        return yp, mu, logvar

    def sample_prior(self, x, act, T=None):
        """
        x: seq_len*bs*dim
        """
        if T is None:
            T = self.horizon
        _, bs, _ = x.shape
        z = torch.randn([bs, self.nh_emb],dtype=x.dtype,device=x.device)
        yp = self.decode(x, z, act, T)
        return yp

class Tranformer_VAE_v3_2(nn.Module):
    """
    decode with action label
    v3_3 + batch norm
    act in posterior
    """

    def __init__(self, nx, ny, horizon, specs):
        super(Tranformer_VAE_v3_2, self).__init__()
        self.nx = nx
        self.ny = ny
        # self.nz = nz
        self.horizon = horizon

        self.n_head = specs.get('n_head', 4)
        self.n_layer = specs.get('n_layer', 8)
        self.p_do = specs.get('p_do', 0.1)
        self.nh_lin = specs.get('nh_lin', 1024)
        self.nh_emb = specs.get('nh_emb', 256)
        self.n_action = n_action = specs.get('n_action', 15)
        self.activation = specs.get('activation', 'gelu')
        self.dec_act = specs.get('decode_act', False)

        enc_layer = nn.TransformerEncoderLayer(d_model=self.nh_emb,
                                               nhead=self.n_head,
                                               dim_feedforward=self.nh_lin,
                                               activation=self.activation)

        # posterior
        self.mu_std_token = nn.Linear(self.n_action, self.nh_emb * 2, bias=False)
        self.enc_lin_proj = nn.Linear(self.nx, self.nh_emb)
        self.enc_transformer = nn.TransformerEncoder(encoder_layer=enc_layer,
                                                     num_layers=self.n_layer)

        # prior
        self.p_mu_std_token = nn.Linear(self.n_action, self.nh_emb * 2, bias=False)
        self.p_enc_lin_proj = nn.Linear(self.nx, self.nh_emb)
        self.p_enc_transformer = nn.TransformerEncoder(encoder_layer=enc_layer,
                                                     num_layers=self.n_layer)

        # decoder
        if self.dec_act:
            self.dec_act_token = nn.Linear(self.n_action, self.nh_emb, bias=True)
        else:
            self.dec_act_token = None
        self.dec_lin_proj_in = nn.Linear(self.nx,self.nh_emb)
        self.dec_lin_proj_out = nn.Linear(self.nh_emb,self.nx)
        self.dec_transformer = CustomTransformerEncoder_v2(encoder_layer=enc_layer,num_layers=self.n_layer,
                                                           cond_dim=self.nh_emb,d_model=self.nh_emb)

        self.pe = PositionalEncoding(d_model=self.nh_emb)


    def encode(self, x, y, act,fn_mask):
        """
        y: seq_len*bs*dim (with history)
        fn_mask: seq_len*bs
        act: bs * naction
        """
        y = torch.cat([x,y],dim=0)
        seq_len, bs, _ = y.shape
        hact = self.mu_std_token(act)  # bs*2nh
        hact = hact.reshape([bs, -1, self.nh_emb]).transpose(0, 1)
        hy = self.enc_lin_proj(y)
        hy = torch.cat([hact, hy], dim=0)
        hy = self.pe(hy)

        # extend mask
        fn_mask = torch.cat([torch.ones_like(fn_mask[:, :2]),
                             torch.ones_like(x[:,:,0].transpose(0,1)),
                             fn_mask],dim=1)==0

        hmu = self.enc_transformer(hy.contiguous(), src_key_padding_mask=fn_mask.contiguous())[:2]
        mu = hmu[0]
        log_var = hmu[1]


        hact = self.p_mu_std_token(act)  # bs*2nh
        hact = hact.reshape([bs, -1, self.nh_emb]).transpose(0, 1)
        hx = self.p_enc_lin_proj(x)
        hx = torch.cat([hact, hx], dim=0)
        hx = self.pe(hx)
        phmu = self.p_enc_transformer(hx.contiguous())[:2]
        pmu = phmu[0]
        plog_var = phmu[1]


        return mu, log_var, pmu, plog_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z, act, T,):
        """
        x: seq_len*bs*dim (only history)
        z: 1*bs*dim
        hx: 1*bs*dim
        """
        if self.dec_act_token:
            hact = self.dec_act_token(act)
            hact = torch.cat([z[None,:,:],hact[None,:,:]],dim=0)
        else:
            hact = z[None,:,:] # 1 * bs * nh
        hx = self.dec_lin_proj_in(x)
        idx = list(range(hx.shape[0]))
        idx = idx + idx[-1:]*T
        ht = hx[idx,:,:]
        # ht = torch.cat([hact[None,:,:],hx],dim=0)
        ht = self.pe(ht)
        # hact = hact[None,:,:].repeat([ht.shape[0],1,1])
        mask_tmp1 = torch.zeros([x.shape[0]+hact.shape[0],ht.shape[0]+hact.shape[0]],device=ht.device)
        mask_tmp1[:,:x.shape[0]+hact.shape[0]] = 1
        mask_tmp2 = torch.tril(torch.ones([T,ht.shape[0]+hact.shape[0]],device=ht.device),
                              diagonal=x.shape[0]+hact.shape[0])
        attn_mask = torch.cat([mask_tmp1,mask_tmp2],dim=0)
        ht = self.dec_transformer(ht, mask=attn_mask==0, cond=hact)[-T:]
        y = self.dec_lin_proj_out(ht)
        y = y + x[-1:]
        # y = []
        # for i in range(T):
        #     htt = self.pe(ht)
        #     hy = self.dec_transformer(htt)[-1:]
        #     ht = torch.cat([ht,hy.detach()],dim=0)
        #     yt = self.dec_lin_proj_out(hy)
        #     y.append(yt)
        #     print(i)
        # y = torch.cat(yt,dim=0)
        return y

    def forward(self, x, y, act, fn_mask):

        mu, logvar, pmu, plogvar = self.encode(x,y, act, fn_mask)
        z = self.reparameterize(mu, logvar) if self.training else mu
        yp = self.decode(x, z, act, y.shape[0])
        return yp, mu, logvar, pmu, plogvar

    def sample_prior(self, x, act, T=None):
        """
        x: seq_len*bs*dim
        """
        if T is None:
            T = self.horizon
        _, bs, _ = x.shape
        hact = self.p_mu_std_token(act)  # bs*2nh
        hact = hact.reshape([bs, -1, self.nh_emb]).transpose(0, 1)
        hx = self.p_enc_lin_proj(x)
        hx = torch.cat([hact, hx], dim=0)
        hx = self.pe(hx)
        phmu = self.p_enc_transformer(hx.contiguous())[:2]
        pmu = phmu[0]
        plog_var = phmu[1]
        z = self.reparameterize(pmu, plog_var)
        yp = self.decode(x, z, act, T)
        return yp


class Tranformer_VAE_v3_2_1(nn.Module):
    """
    decode with action label
    v3_3 + batch norm
    act in posterior
    """

    def __init__(self, nx, ny, horizon, specs):
        super(Tranformer_VAE_v3_2_1, self).__init__()
        self.nx = nx
        self.ny = ny
        # self.nz = nz
        self.horizon = horizon

        self.n_head = specs.get('n_head', 4)
        self.n_layer = specs.get('n_layer', 8)
        self.p_do = specs.get('p_do', 0.1)
        self.nh_lin = specs.get('nh_lin', 1024)
        self.nh_emb = specs.get('nh_emb', 256)
        self.n_action = n_action = specs.get('n_action', 15)
        self.activation = specs.get('activation', 'gelu')
        self.dec_act = specs.get('decode_act', False)

        enc_layer = nn.TransformerEncoderLayer(d_model=self.nh_emb,
                                               nhead=self.n_head,
                                               dim_feedforward=self.nh_lin,
                                               activation=self.activation)

        # posterior
        self.mu_std_token = nn.Linear(self.n_action, self.nh_emb * 2, bias=False)
        self.enc_lin_proj = nn.Linear(self.nx, self.nh_emb)
        self.enc_transformer = nn.TransformerEncoder(encoder_layer=enc_layer,
                                                     num_layers=self.n_layer)

        # prior
        # self.p_mu_std_token = nn.Linear(self.n_action, self.nh_emb * 2, bias=False)
        # self.p_enc_lin_proj = nn.Linear(self.nx, self.nh_emb)
        # self.p_enc_transformer = nn.TransformerEncoder(encoder_layer=enc_layer,
        #                                              num_layers=self.n_layer)

        # decoder
        if self.dec_act:
            self.dec_act_token = nn.Linear(self.n_action, self.nh_emb, bias=True)
        else:
            self.dec_act_token = None
        self.dec_lin_proj_in = nn.Linear(self.nx,self.nh_emb)
        self.dec_lin_proj_out = nn.Linear(self.nh_emb,self.nx)
        self.dec_transformer = CustomTransformerEncoder_v2(encoder_layer=enc_layer,num_layers=self.n_layer,
                                                           cond_dim=self.nh_emb,d_model=self.nh_emb)

        self.pe = PositionalEncoding(d_model=self.nh_emb)


    def encode(self, x, y, act,fn_mask):
        """
        y: seq_len*bs*dim (with history)
        fn_mask: seq_len*bs
        act: bs * naction
        """
        y = torch.cat([x,y],dim=0)
        seq_len, bs, _ = y.shape
        hact = self.mu_std_token(act)  # bs*2nh
        hact = hact.reshape([bs, -1, self.nh_emb]).transpose(0, 1)
        hy = self.enc_lin_proj(y)
        hy = torch.cat([hact, hy], dim=0)
        hy = self.pe(hy)

        # extend mask
        fn_mask = torch.cat([torch.ones_like(fn_mask[:, :2]),
                             torch.ones_like(x[:,:,0].transpose(0,1)),
                             fn_mask],dim=1)==0

        hmu = self.enc_transformer(hy.contiguous(), src_key_padding_mask=fn_mask.contiguous())[:2]
        mu = hmu[0]
        log_var = hmu[1]


        # hact = self.p_mu_std_token(act)  # bs*2nh
        # hact = hact.reshape([bs, -1, self.nh_emb]).transpose(0, 1)
        # hx = self.p_enc_lin_proj(x)
        # hx = torch.cat([hact, hx], dim=0)
        # hx = self.pe(hx)
        # phmu = self.p_enc_transformer(hx.contiguous())[:2]
        # pmu = phmu[0]
        # plog_var = phmu[1]


        return mu, log_var, 0, 0

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z, act, T,):
        """
        x: seq_len*bs*dim (only history)
        z: 1*bs*dim
        hx: 1*bs*dim
        """
        if self.dec_act_token:
            hact = self.dec_act_token(act)
            hact = torch.cat([z[None,:,:],hact[None,:,:]],dim=0)
        else:
            hact = z[None,:,:] # 1 * bs * nh
        hx = self.dec_lin_proj_in(x)
        idx = list(range(hx.shape[0]))
        idx = idx + idx[-1:]*T
        ht = hx[idx,:,:]
        # ht = torch.cat([hact[None,:,:],hx],dim=0)
        ht = self.pe(ht)
        # hact = hact[None,:,:].repeat([ht.shape[0],1,1])
        mask_tmp1 = torch.zeros([x.shape[0]+hact.shape[0],ht.shape[0]+hact.shape[0]],device=ht.device)
        mask_tmp1[:,:x.shape[0]+hact.shape[0]] = 1
        mask_tmp2 = torch.tril(torch.ones([T,ht.shape[0]+hact.shape[0]],device=ht.device),
                              diagonal=x.shape[0]+hact.shape[0])
        attn_mask = torch.cat([mask_tmp1,mask_tmp2],dim=0)
        ht = self.dec_transformer(ht, mask=attn_mask==0, cond=hact)[-T:]
        y = self.dec_lin_proj_out(ht)
        y = y + x[-1:]
        # y = []
        # for i in range(T):
        #     htt = self.pe(ht)
        #     hy = self.dec_transformer(htt)[-1:]
        #     ht = torch.cat([ht,hy.detach()],dim=0)
        #     yt = self.dec_lin_proj_out(hy)
        #     y.append(yt)
        #     print(i)
        # y = torch.cat(yt,dim=0)
        return y

    def forward(self, x, y, act, fn_mask):

        mu, logvar, pmu, plogvar = self.encode(x,y, act, fn_mask)
        z = self.reparameterize(mu, logvar) if self.training else mu
        yp = self.decode(x, z, act, y.shape[0])
        return yp, mu, logvar, pmu, plogvar

    def sample_prior(self, x, act, T=None):
        """
        x: seq_len*bs*dim
        """
        if T is None:
            T = self.horizon
        _, bs, _ = x.shape
        # hact = self.p_mu_std_token(act)  # bs*2nh
        # hact = hact.reshape([bs, -1, self.nh_emb]).transpose(0, 1)
        # hx = self.p_enc_lin_proj(x)
        # hx = torch.cat([hact, hx], dim=0)
        # hx = self.pe(hx)
        # phmu = self.p_enc_transformer(hx.contiguous())[:2]
        # pmu = phmu[0]
        # plog_var = phmu[1]
        # z = self.reparameterize(pmu, plog_var)
        z = torch.randn([x.shape[1],self.nh_emb]).to(dtype=x.dtype,device=x.device)
        yp = self.decode(x, z, act, T)
        return yp

class Tranformer_VAE_v3_3(nn.Module):
    """
    decode with action label
    v3_3 + batch norm
    act in posterior
    """

    def __init__(self, nx, ny, horizon, specs):
        super(Tranformer_VAE_v3_3, self).__init__()
        self.nx = nx
        self.ny = ny
        # self.nz = nz
        self.horizon = horizon

        self.n_head = specs.get('n_head', 4)
        self.n_layer = specs.get('n_layer', 8)
        self.p_do = specs.get('p_do', 0.1)
        self.nh_lin = specs.get('nh_lin', 1024)
        self.nh_emb = specs.get('nh_emb', 256)
        self.n_action = n_action = specs.get('n_action', 15)
        self.activation = specs.get('activation', 'gelu')
        self.dec_act = specs.get('decode_act', False)

        enc_layer = nn.TransformerEncoderLayer(d_model=self.nh_emb,
                                               nhead=self.n_head,
                                               dim_feedforward=self.nh_lin,
                                               activation=self.activation)

        # posterior
        self.mu_std_token = nn.Linear(self.n_action, self.nh_emb * 2, bias=False)
        self.enc_lin_proj = nn.Linear(self.nx, self.nh_emb)
        self.enc_transformer = nn.TransformerEncoder(encoder_layer=enc_layer,
                                                     num_layers=self.n_layer)

        # prior
        self.p_mu_std_token = nn.Linear(self.n_action, self.nh_emb * 2, bias=False)
        self.p_enc_lin_proj = nn.Linear(self.nx, self.nh_emb)
        self.p_enc_transformer = nn.TransformerEncoder(encoder_layer=enc_layer,
                                                     num_layers=self.n_layer)

        # decoder
        if self.dec_act:
            self.dec_act_token = nn.Linear(self.n_action, self.nh_emb, bias=True)
        else:
            self.dec_act_token = None
        self.dec_lin_proj_in = nn.Linear(self.nx,self.nh_emb)
        self.dec_lin_proj_out = nn.Linear(self.nh_emb,self.nx)
        self.dec_stopsign_out = nn.Sequential(nn.Linear(self.nh_emb,1), nn.Sigmoid())
        self.dec_transformer = CustomTransformerEncoder_v2(encoder_layer=enc_layer,num_layers=self.n_layer,
                                                           cond_dim=self.nh_emb,d_model=self.nh_emb)

        self.pe = PositionalEncoding(d_model=self.nh_emb)


    def encode(self, x, y, act,fn_mask):
        """
        y: seq_len*bs*dim (with history)
        fn_mask: seq_len*bs
        act: bs * naction
        """
        y = torch.cat([x,y],dim=0)
        seq_len, bs, _ = y.shape
        hact = self.mu_std_token(act)  # bs*2nh
        hact = hact.reshape([bs, -1, self.nh_emb]).transpose(0, 1)
        hy = self.enc_lin_proj(y)
        hy = torch.cat([hact, hy], dim=0)
        hy = self.pe(hy)

        # extend mask
        fn_mask = torch.cat([torch.ones_like(fn_mask[:, :2]),
                             torch.ones_like(x[:,:,0].transpose(0,1)),
                             fn_mask],dim=1)==0

        hmu = self.enc_transformer(hy.contiguous(), src_key_padding_mask=fn_mask.contiguous())[:2]
        mu = hmu[0]
        log_var = hmu[1]


        hact = self.p_mu_std_token(act)  # bs*2nh
        hact = hact.reshape([bs, -1, self.nh_emb]).transpose(0, 1)
        hx = self.p_enc_lin_proj(x)
        hx = torch.cat([hact, hx], dim=0)
        hx = self.pe(hx)
        phmu = self.p_enc_transformer(hx.contiguous())[:2]
        pmu = phmu[0]
        plog_var = phmu[1]


        return mu, log_var, pmu, plog_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z, act, T,):
        """
        x: seq_len*bs*dim (only history)
        z: 1*bs*dim
        hx: 1*bs*dim
        """
        if self.dec_act_token:
            hact = self.dec_act_token(act)
            hact = torch.cat([z[None,:,:],hact[None,:,:]],dim=0)
        else:
            hact = z[None,:,:] # 1 * bs * nh
        hx = self.dec_lin_proj_in(x)
        idx = list(range(hx.shape[0]))
        idx = idx + idx[-1:]*T
        ht = hx[idx,:,:]
        # ht = torch.cat([hact[None,:,:],hx],dim=0)
        ht = self.pe(ht)
        # hact = hact[None,:,:].repeat([ht.shape[0],1,1])
        mask_tmp1 = torch.zeros([x.shape[0]+hact.shape[0],ht.shape[0]+hact.shape[0]],device=ht.device)
        mask_tmp1[:,:x.shape[0]+hact.shape[0]] = 1
        mask_tmp2 = torch.tril(torch.ones([T,ht.shape[0]+hact.shape[0]],device=ht.device),
                              diagonal=x.shape[0]+hact.shape[0])
        attn_mask = torch.cat([mask_tmp1,mask_tmp2],dim=0)
        ht = self.dec_transformer(ht, mask=attn_mask==0, cond=hact)[-T:]
        y = self.dec_lin_proj_out(ht)
        y = y + x[-1:]
        ss = self.dec_stopsign_out(ht)[:,:,0].transpose(0,1)
        # y = []
        # for i in range(T):
        #     htt = self.pe(ht)
        #     hy = self.dec_transformer(htt)[-1:]
        #     ht = torch.cat([ht,hy.detach()],dim=0)
        #     yt = self.dec_lin_proj_out(hy)
        #     y.append(yt)
        #     print(i)
        # y = torch.cat(yt,dim=0)
        return y, ss

    def forward(self, x, y, act, fn_mask):

        mu, logvar, pmu, plogvar = self.encode(x,y, act, fn_mask)
        z = self.reparameterize(mu, logvar) if self.training else mu
        yp, ss = self.decode(x, z, act, y.shape[0])
        return yp, ss, mu, logvar, pmu, plogvar

    def sample_prior(self, x, act, T=None):
        """
        x: seq_len*bs*dim
        """
        if T is None:
            T = self.horizon
        _, bs, _ = x.shape
        hact = self.p_mu_std_token(act)  # bs*2nh
        hact = hact.reshape([bs, -1, self.nh_emb]).transpose(0, 1)
        hx = self.p_enc_lin_proj(x)
        hx = torch.cat([hact, hx], dim=0)
        hx = self.pe(hx)
        phmu = self.p_enc_transformer(hx.contiguous())[:2]
        pmu = phmu[0]
        plog_var = phmu[1]
        z = self.reparameterize(pmu, plog_var)
        yp, ss = self.decode(x, z, act, T)
        return yp, ss

class Tranformer_VAE_v4(nn.Module):
    """
    decode with action label
    v3_3 + batch norm
    act in posterior
    """

    def __init__(self, nx, ny, horizon, specs):
        super(Tranformer_VAE_v4, self).__init__()
        self.nx = nx
        self.ny = ny
        # self.nz = nz
        self.horizon = horizon

        self.n_head = specs.get('n_head', 4)
        self.n_layer = specs.get('n_layer', 8)
        self.p_do = specs.get('p_do', 0.1)
        self.nh_lin = specs.get('nh_lin', 1024)
        self.nh_emb = specs.get('nh_emb', 256)
        self.n_action = n_action = specs.get('n_action', 15)
        self.activation = specs.get('activation', 'gelu')

        enc_layer = nn.TransformerEncoderLayer(d_model=self.nh_emb,
                                               nhead=self.n_head,
                                               dim_feedforward=self.nh_lin,
                                               activation=self.activation)

        # history encoder
        self.his_lin_proj = nn.Linear(self.nx, self.nh_emb)
        self.enc_x = nn.TransformerEncoder(encoder_layer=enc_layer,
                                           num_layers=self.n_layer//2)


        # posterior
        self.mu_std_token = nn.Linear(self.n_action, self.nh_emb * 2, bias=False)
        self.enc_lin_proj = nn.Linear(self.nx, self.nh_emb)
        self.enc_transformer = CustomTransformerEncoder_v2(encoder_layer=enc_layer,
                                                           num_layers=self.n_layer,
                                                           cond_dim=self.nh_emb,
                                                           d_model=self.nh_emb)


        # decoder
        self.dec_act_token = nn.Linear(self.n_action, self.nh_emb, bias=False)
        # self.dec_lin_proj_in = nn.Linear(self.nx,self.nh_emb)
        self.dec_lin_proj_out = nn.Linear(self.nh_emb,self.nx)
        self.dec_transformer = CustomTransformerEncoder_v2(encoder_layer=enc_layer,num_layers=self.n_layer,
                                                           cond_dim=self.nh_emb,d_model=self.nh_emb)

        self.pe = PositionalEncoding(d_model=self.nh_emb)


    def encode(self, x, y, act, fn_mask):
        """
        x: t_his*bs*dim
        y: t_pre*bs*dim
        fn_mask: seq_len*bs
        act: bs * naction
        """
        hx = self.his_lin_proj(x)
        hx = self.enc_x(hx)

        seq_len, bs, _ = y.shape
        hact = self.mu_std_token(act)  # bs*2nh
        hact = hact.reshape([bs, -1, self.nh_emb]).transpose(0, 1)
        hy = self.enc_lin_proj(y)
        hy = torch.cat([hact, hy], dim=0)
        hy = self.pe(hy)

        # idx = [0] * 2 + list(range(seq_len))
        fn_tmp = torch.zeros_like(fn_mask[:, :2+x.shape[0]])==1
        fn_mask = torch.cat([fn_tmp,fn_mask],dim=1)

        hmu = self.enc_transformer(hy.contiguous(), cond=hx.contiguous(),
                                   src_key_padding_mask=fn_mask.contiguous())[:2]
        mu = hmu[0]
        log_var = hmu[1]

        return mu, log_var, hx

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z, act, T, hx=None):
        """
        x: seq_len*bs*dim (only history)
        z: 1*bs*dim
        hx: 1*bs*dim
        """
        bs= x.shape[1]

        if hx is None:
            hx = self.his_lin_proj(x)
            hx = self.enc_x(hx)

        hact = self.dec_act_token(act)
        hact = z + hact # 1 * bs * nh
        hcond = torch.cat([hact[None,:,:],hx],dim=0)
        # hx = self.dec_lin_proj_in(x)
        # idx = list(range(hx.shape[0]))
        # idx = idx + idx[-1:]*T
        # ht = hx[idx,:,:]
        # ht = torch.cat([hact[None,:,:],hx],dim=0)
        ht = torch.zeros([T,bs,self.nh_emb],dtype=x.dtype,device=x.device)
        ht = self.pe(ht)
        # hact = hact[None,:,:].repeat([ht.shape[0],1,1])
        mask_tmp1 = torch.zeros([hcond.shape[0],hcond.shape[0]+T],device=hcond.device)
        mask_tmp1[:,:hcond.shape[0]] = 1
        mask_tmp2 = torch.tril(torch.ones([T,hcond.shape[0]+T],device=hcond.device),
                              diagonal=hcond.shape[0]-1)
        attn_mask = torch.cat([mask_tmp1,mask_tmp2],dim=0)
        ht = self.dec_transformer(ht, mask=attn_mask==0, cond=hcond)
        y = self.dec_lin_proj_out(ht)
        y = y + x[-1:]
        # y = []
        # for i in range(T):
        #     htt = self.pe(ht)
        #     hy = self.dec_transformer(htt)[-1:]
        #     ht = torch.cat([ht,hy.detach()],dim=0)
        #     yt = self.dec_lin_proj_out(hy)
        #     y.append(yt)
        #     print(i)
        # y = torch.cat(yt,dim=0)
        return y

    def forward(self, x, y, act, fn_mask):
        # if fn_mask.shape[1] == y.shape[0]:
        #     # extend mask
        #     fn_mask = torch.cat([torch.zeros_like(x[:,:,0].transpose(0,1))==1,
        #                          fn_mask],dim=1)

        mu, logvar, hx = self.encode(x, y, act, fn_mask)
        z = self.reparameterize(mu, logvar) if self.training else mu
        yp = self.decode(x, z, act, y.shape[0], hx=hx)
        return yp, mu, logvar

    def sample_prior(self, x, act, T=None):
        """
        x: seq_len*bs*dim
        """
        if T is None:
            T = self.horizon
        _, bs, _ = x.shape
        z = torch.randn([bs, self.nh_emb],dtype=x.dtype,device=x.device)
        yp = self.decode(x, z, act, T)
        return yp


class Tranformer_VAE_v5(nn.Module):
    """
    decode with action label
    v3_3 + batch norm
    act in posterior
    """

    def __init__(self, nx, ny, horizon, specs):
        super(Tranformer_VAE_v5, self).__init__()
        self.nx = nx
        self.ny = ny
        # self.nz = nz
        self.horizon = horizon

        self.n_head = specs.get('n_head', 4)
        self.n_layer = specs.get('n_layer', 8)
        self.p_do = specs.get('p_do', 0.1)
        self.nh_lin = specs.get('nh_lin', 1024)
        self.nh_emb = nh_emb = specs.get('nh_emb', 256)
        self.nz = nz = specs.get('nz', 128)
        self.n_action = n_action = specs.get('n_action', 15)
        self.activation = specs.get('activation', 'gelu')
        self.nh_mlp = nh_mlp = specs.get('nh_mlp', [300, 200])

        enc_layer = nn.TransformerEncoderLayer(d_model=self.nh_emb,
                                               nhead=self.n_head,
                                               dim_feedforward=self.nh_lin,
                                               activation=self.activation)

        # x encoder
        self.x_enc_lin_proj = nn.Linear(self.nx, self.nh_emb)
        self.x_feat_token = nn.Parameter(torch.randn(1, 1, self.nh_emb))
        # init:
        stdv = 1. / math.sqrt(self.nh_emb)
        self.x_feat_token.data.uniform_(-stdv, stdv)
        self.x_enc_transformer = nn.TransformerEncoder(encoder_layer=enc_layer,
                                                       num_layers=self.n_layer)

        # y encoder
        self.y_enc_lin_proj = nn.Linear(self.nx, self.nh_emb)
        self.y_feat_token = nn.Parameter(torch.randn(1, 1, self.nh_emb))
        # init:
        stdv = 1. / math.sqrt(self.nh_emb)
        self.y_feat_token.data.uniform_(-stdv, stdv)
        self.y_enc_transformer = nn.TransformerEncoder(encoder_layer=enc_layer,
                                                       num_layers=self.n_layer)

        # posterior
        self.e_mlp = MLP(3 * nh_emb, nh_mlp, is_bn=True)
        self.e_mu = nn.Linear(self.e_mlp.out_dim, nz)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz)

        # prior
        self.p_act_mlp = nn.Linear(n_action, nh_emb)
        self.p_mlp = MLP(2 * self.nh_emb, nh_mlp, is_bn=True)
        self.p_mu = nn.Linear(self.p_mlp.out_dim, self.nh_emb)
        self.p_logvar = nn.Linear(self.p_mlp.out_dim, self.nh_emb)

        # decoder
        self.d_mlp_in = MLP(nh_emb*2+nz, nh_mlp)
        self.d_in = nn.Linear(self.d_mlp_in.out_dim, nh_emb)
        self.d_mlp_out = MLP(nh_emb, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp_out.out_dim, ny)
        self.dec_transformer = nn.TransformerEncoder(encoder_layer=enc_layer,
                                                       num_layers=self.n_layer)
        self.pe = PositionalEncoding(d_model=self.nh_emb)

    def encode_x(self,x):
        """
        x: seq,bs,feat
        """
        feat_token = self.x_feat_token.repeat([1,x.shape[1],1])
        hx = self.x_enc_lin_proj(x)
        hx = torch.cat([feat_token,hx],dim=0)
        hx = self.pe(hx)
        hx = self.x_enc_transformer(hx)[0]
        return hx

    def encode_y(self,y, fn_mask):
        """
        x: seq,bs,feat
        """
        feat_token = self.y_feat_token.repeat([1,y.shape[1],1])
        hy = self.y_enc_lin_proj(y)
        hy = torch.cat([feat_token,hy],dim=0)
        hy = self.pe(hy)
        tmp = torch.ones_like(fn_mask[:,:1])
        fn_mask = torch.cat([tmp,fn_mask],dim=1)==0
        hy = self.y_enc_transformer(hy,src_key_padding_mask=fn_mask.contiguous())[0]
        return hy

    def encode(self, x, y, act, fn_mask):
        """
        y: seq_len*bs*dim (with history)
        fn_mask: seq_len*bs
        act: bs * naction
        """
        hx = self.encode_x(x)
        hy = self.encode_y(y,fn_mask)
        hact = self.p_act_mlp(act)

        h = torch.cat((hx, hy, hact), dim=1)
        h = self.e_mlp(h)
        emu = self.e_mu(h)
        elogvar = self.e_logvar(h)

        h = torch.cat((hx, hact), dim=1)
        h = self.p_mlp(h)
        pmu = self.p_mu(h)
        plogvar = self.p_logvar(h)

        return emu, elogvar, pmu, plogvar, hx, hact

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z, act, T, hx=None,hact=None):
        """
        x: seq_len*bs*dim (only history)
        z: 1*bs*dim
        hx: 1*bs*dim
        """
        if hx is None:
            hx = self.encode_x(x)
        if hact is None:
            hact = self.p_act_mlp(act)

        ht = torch.cat([z,hx,hact],dim=1)
        ht = self.d_mlp_in(ht)
        ht = self.d_in(ht)
        ht = ht[None,:,:].repeat([T,1,1])
        ht = self.pe(ht)
        # hact = hact[None,:,:].repeat([ht.shape[0],1,1])
        attn_mask = torch.tril(torch.ones([T,T],device=ht.device),diagonal=0) == 0
        ht = self.dec_transformer(ht, mask=attn_mask)
        ht = self.d_mlp_out(ht)
        y = self.d_out(ht)
        y = y + x[-1:]
        return y

    def forward(self, x, y, act, fn_mask):
        mu, logvar, pmu, plogvar, hx, hact = self.encode(x, y, act, fn_mask)
        z = self.reparameterize(mu, logvar) if self.training else mu
        yp = self.decode(x, z, act, y.shape[0], hx, hact)
        return yp, mu, logvar, pmu, plogvar

    def sample_prior(self, x, act, T=None):
        """
        x: seq_len*bs*dim
        """
        if T is None:
            T = self.horizon
        _, bs, _ = x.shape
        z = torch.randn([bs, self.nz],dtype=x.dtype,device=x.device)
        yp = self.decode(x, z, act, T)
        return yp
