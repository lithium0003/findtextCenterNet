import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import itertools

from util_func import modulo_list, calc_predid, feature_dim
from const import decoder_SOT, decoder_EOT, decoder_MSK, max_decoderlen, max_encoderlen, encoder_add_dim

encoder_dim = feature_dim + encoder_add_dim

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len = 5000):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super().__init__()

        # same size with input matrix (for adding with input matrix)
        encoding = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model / 4)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model / 4)))
        # compute positional encoding to consider positional information of words

        self.encoding = nn.Buffer(encoding).requires_grad_(False)
        # self.encoding = nn.Parameter(encoding)

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        seq_len = x.shape[1]
        # [batch_size = 128, seq_len = 30]

        pe = self.encoding[:seq_len, :].unsqueeze(0)
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]

        return x + pe.type_as(x)

class SwiGLU(nn.Module):
    def __init__(self, dim, dropout = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, dim*2)
        self.wg = nn.Linear(dim, dim*2)
        self.w2 = nn.Linear(dim*2, dim)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        x1 = self.w1(x)
        xg = F.silu(self.wg(x))
        x = x1 * xg
        x = self.dropout(x)
        return self.w2(x)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x.contiguous()
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

class ScaleUp(nn.Module):
    ### Learned pararmeter used to scale up QKt before taking the softmax
    """ScaleUp"""
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale).float())

    def forward(self, x):
        return x * self.scale

class MultiheadAttn(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout = 0.1,
        max_seq_len=5000,
        seq_len_threshold=72,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # arg num_heads set to half of Transformer's num_heads
        self.num_heads = num_heads
        
        # arg decoder_kv_attention_heads set to half of Transformer's num_kv_heads if use GQA
        # set to same as num_heads if use normal MHA
        self.num_kv_heads = num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = embed_dim // num_heads
        self.seq_len_threshold = seq_len_threshold
        self.scaling = self.head_dim ** -0.5
        self.mha_scale = ScaleUp(np.log2(self.seq_len_threshold**2 - self.seq_len_threshold))
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.pos_emb_q = PositionalEncoding(embed_dim, max_len=max_seq_len)
        self.pos_emb_k = PositionalEncoding(embed_dim, max_len=max_seq_len)

        self.dropout = nn.Dropout(p = dropout, inplace=True)

    def forward(
        self,
        query, key=None, value=None,
        causal_mask=None,
        key_mask=None,
    ):
        if key is None:
            key = query
            pos_emb_k = self.pos_emb_q
        else:
            pos_emb_k = self.pos_emb_k
        if value is None:
            value = key
        bsz, tgt_len, embed_dim = query.size()
        bsz, src_len, embed_dim = key.size()

        query = self.pos_emb_q(query)
        key = pos_emb_k(key)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2).contiguous()
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)

        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))

        if key_mask is not None:
            attn_weights += key_mask[:,:,:,:src_len].type_as(attn_weights)
        if causal_mask is not None:
            attn_weights += causal_mask[:tgt_len,:src_len].type_as(attn_weights)
        attn_weights = F.softmax(attn_weights.float(), dim=-1, dtype=torch.float32).type_as(attn_weights)

        attn_weights = self.dropout(attn_weights)

        attn = torch.matmul(attn_weights, v)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * self.head_dim)

        attn = self.out_proj(attn)
        return attn

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, dropout = 0.1, max_seq_len=5000):
        super().__init__()
        self.mha = MultiheadAttn(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, max_seq_len=max_seq_len)
        self.norm1 = nn.LayerNorm([embed_dim])
        self.norm2 = nn.LayerNorm([embed_dim])
        self.ff = SwiGLU(embed_dim, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, key_mask=None):
        skip = x
        x = self.mha(x, key_mask=key_mask)
        x = self.dropout1(x)
        x = x + skip
        x = self.norm1(x)
        _x = x
        x = self.ff(x)
        x = self.dropout2(x)
        x = x + _x + skip
        x = self.norm2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, head_num, max_seq_len=5000, block_num = 6, dropout = 0.1):
        super().__init__()
        self.dim = embed_dim
        self.head_num = head_num
        self.embed = nn.Linear(input_dim, embed_dim, bias=False)
        self.pos_emb = PositionalEncoding(embed_dim, max_len=max_seq_len)
        self.norm = nn.LayerNorm([embed_dim])
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlock(embed_dim, d, head_num, dropout=dropout, max_seq_len=max_seq_len) for d in range(block_num)])        

    def forward(self, x, key_mask=None):
        x = self.embed(x)
        x = self.pos_emb(x)
        x = self.norm(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, key_mask=key_mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, depth, head_num, dropout = 0.1, max_seq_len=5000):
        super().__init__()
        self.self_attn = MultiheadAttn(embed_dim, head_num, dropout=dropout, max_seq_len=max_seq_len)
        self.cross_attn = MultiheadAttn(embed_dim, head_num, dropout=dropout, max_seq_len=max_seq_len)
        self.norm1 = nn.LayerNorm([embed_dim])
        self.norm2 = nn.LayerNorm([embed_dim])
        self.norm3 = nn.LayerNorm([embed_dim])
        self.ff = SwiGLU(embed_dim, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, y, causal_mask=None, key_mask=None):
        skip = x
        x = self.self_attn(x, causal_mask=causal_mask)
        x = self.dropout1(x)
        x = x + skip
        x = self.norm1(x)
        _x = x
        x = self.cross_attn(x, y, key_mask=key_mask)
        x = self.dropout2(x)
        x = x + _x
        x = self.norm2(x)
        _x = x
        x = self.ff(x)
        x = self.dropout3(x)
        x = x + _x + skip
        x = self.norm3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embed_dim, head_num, max_seq_len=5000, block_num = 6, dropout = 0.1):
        super().__init__()
        self.head_num = head_num
        self.max_seq_len = max_seq_len
        self.embed = nn.ModuleList([nn.Embedding(m, embed_dim) for m in modulo_list])
        self.pos_emb = PositionalEncoding(embed_dim, max_len=max_seq_len)
        self.norm = nn.LayerNorm([embed_dim])
        self.blocks = nn.ModuleList([DecoderBlock(embed_dim, d, head_num, dropout=dropout, max_seq_len=max_seq_len) for d in range(block_num)])
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.out_layers = nn.ModuleList([nn.Linear(embed_dim, m) for m in modulo_list])

    def forward(self, x, y, causal_mask=None, key_mask=None):
        x1 = [x % m for m in modulo_list]
        x = None
        for x2, layer in zip(x1, self.embed):
            if x is None:
                x = layer(x2)
            else:
                x += layer(x2)
        x = self.pos_emb(x)
        x = self.norm(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, y, causal_mask=causal_mask, key_mask=key_mask)
        return [layer(x) for layer in self.out_layers]

class Transformer(nn.Module):
    def __init__(self, enc_input_dim, embed_dim, head_num, enc_block_num = 6, dec_block_num = 6, max_enc_seq_len = 5000, max_dec_seq_len = 5000, dropout = 0.1):
        super().__init__()
        self.head_num = head_num
        self.max_len = max(max_enc_seq_len, max_dec_seq_len)
        self.encoder = Encoder(input_dim=enc_input_dim, embed_dim=embed_dim, head_num=head_num, max_seq_len=max_enc_seq_len, block_num=enc_block_num, dropout=dropout)
        self.decoder = Decoder(embed_dim=embed_dim, head_num=head_num, max_seq_len=max_dec_seq_len, block_num=dec_block_num, dropout=dropout)

    def forward(self, enc_input, dec_input):
        key_mask = torch.all(enc_input == 0, dim=-1)
        key_mask = torch.where(key_mask[:,None,None,:], float("-inf"), 0)
        enc_output = self.encoder(enc_input, key_mask=key_mask)
        output = self.decoder(dec_input, enc_output, key_mask=key_mask)
        return output

@dataclass
class ModelDimensions:
    enc_input_dim: int = encoder_dim
    embed_dim: int = 512
    head_num: int = 16
    enc_block_num: int = 12
    dec_block_num: int = 12
    max_enc_seq_len: int = max_encoderlen
    max_dec_seq_len: int = max_decoderlen
    dropout: float = 0.0

class TransformerPredictor(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.head_num = encoder.head_num
        self.max_len = decoder.max_seq_len
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_input):
        key_mask = torch.all(enc_input == 0, dim=-1)
        key_mask = torch.where(key_mask[:,None,None,:], float("-inf"), 0)
        enc_output = self.encoder(enc_input, key_mask=key_mask)
        decoder_input = torch.zeros((enc_input.shape[0],max_decoderlen), dtype=torch.long, device=enc_input.device)
        decoder_input[:,0] = decoder_SOT
        decoder_input[:,1:] = decoder_MSK
        rep_count = 10
        for k in range(rep_count):
            outputs = self.decoder(decoder_input, enc_output, key_mask=key_mask)
            listp = []
            listi = []
            for decoder_id1 in outputs:
                pred_p1 = torch.softmax(decoder_id1, dim=-1)
                topp, topi = torch.topk(pred_p1, 4)
                listp.append(topp.permute(2,0,1))
                listi.append(topi.permute(2,0,1))

            pred_ids = torch.stack([torch.stack(x) for x in itertools.product(*listi)]).transpose(0,1)
            pred_p = torch.stack([torch.stack(x) for x in itertools.product(*listp)]).transpose(0,1)
            pred_p = pred_p.clamp_min(1e-10).log().mean(dim=0).exp()
            decoder_output = calc_predid(*pred_ids)
            maxi = torch.argmax(pred_p, dim=0)
            pred_p[decoder_output > 0x3FFFF] = 0
            maxi = torch.argmax(pred_p, dim=0)
            decoder_output = torch.gather(decoder_output, 0, maxi.unsqueeze(0))[0]
            pred_p = torch.gather(pred_p, 0, maxi.unsqueeze(0))[0]
            if k > 0 and torch.all(pred_p[decoder_output > 0] > 0.99):
                print(f'[{k} early stop]')
                break
            if k < rep_count-1:
                r = int(max_decoderlen * (k + 1) / rep_count)
                remask = torch.arange(max_decoderlen, device=enc_input.device) > r
                if r > 0:
                    sorted, indices = torch.sort(-pred_p[:,:r])
                    s = int(r / rep_count * (k+1))
                    p_th = sorted[:,s]
                    remask = torch.logical_or(remask, pred_p < p_th)
                if not torch.any(remask):
                    break
                decoder_output = torch.where(remask, decoder_MSK, decoder_output)
                decoder_input[:,1:] = decoder_output[:,:-1]
        return decoder_output

class TransformerEncoderPredictor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.head_num = encoder.head_num
        self.encoder = encoder

    def forward(self, enc_input, key_mask):
        enc_output = self.encoder(enc_input, key_mask)
        return enc_output

class DecoderSplited(Decoder):
    def forward(self, x1, y, causal_mask=None, key_mask=None):
        x = None
        for x2, layer in zip(x1, self.embed):
            if x is None:
                x = layer(x2)
            else:
                x += layer(x2)
        x = self.pos_emb(x)
        x = self.norm(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, y, causal_mask=causal_mask, key_mask=key_mask)
        return [layer(x) for layer in self.out_layers]

class TransformerDecoderPredictor(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.head_num = decoder.head_num
        self.decoder = decoder

    def forward(self, enc_output, decoder_input, key_mask):
        outputs = self.decoder(decoder_input, enc_output, key_mask=key_mask)
        return [torch.softmax(output, dim=-1) for output in outputs]

if __name__ == '__main__':
    model = Transformer(enc_input_dim=100, embed_dim=512, head_num=8)
    # print(model)
    # out = model(torch.ones(3,1,100),torch.ones(3,2, dtype=torch.long))
    # print(out)
    # print([o.shape for o in out])

    # model2 = TransformerPredictor(model.encoder, model.decoder)
    # print(model2)
    # d = model2(torch.ones(4,5,100))
    # print(d)

    TransformerDecoderPredictor(model.decoder)