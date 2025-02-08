import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

from util_func import modulo_list, calc_predid, feature_dim
from const import decoder_SOT, decoder_EOT, max_decoderlen, encoder_add_dim

encoder_dim = feature_dim + encoder_add_dim

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len = 5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, dim))

    def forward(self, x, offset = 0):
        pe = self.pe[offset:offset+x.size(1)].unsqueeze(0)
        return x + pe

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim*8//3)
        self.w2 = nn.Linear(dim*8//3, dim)
        self.wg = nn.Linear(dim, dim*8//3)

    def forward(self, x):
        x1 = self.w1(x)
        xg = F.silu(self.wg(x))
        return self.w2(x1 * xg)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # arg num_heads set to half of Transformer's num_heads
        self.num_heads = num_heads
        
        # arg decoder_kv_attention_heads set to half of Transformer's num_kv_heads if use GQA
        # set to same as num_heads if use normal MHA
        self.num_kv_heads = num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = nn.RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    
    def forward(
        self,
        query, key=None, value=None,
        attn_mask=None,
    ):
        if key is None:
            key = query
        if value is None:
            value = key
        bsz, tgt_len, embed_dim = query.size()
        bsz, src_len, embed_dim = key.size()

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += attn_mask   
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        return attn
    
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, dropout = 0.1):
        super().__init__()
        self.mha = MultiheadDiffAttn(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        self.norm1 = nn.LayerNorm([embed_dim])
        self.norm2 = nn.LayerNorm([embed_dim])
        self.ff = SwiGLU(embed_dim)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)

    def forward(self, x, attn_mask=None):
        skip = x
        x = self.mha(x, attn_mask=attn_mask)
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
        self.embed = nn.Linear(input_dim, embed_dim)
        self.pe = PositionalEncoding(embed_dim, max_len=max_seq_len)
        self.norm = nn.LayerNorm([embed_dim])
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.blocks = nn.ModuleList([EncoderBlock(embed_dim, d, head_num) for d in range(block_num)])        

    def forward(self, x, attn_mask=None, offset=0):
        x = self.embed(x)
        x = self.pe(x,offset=offset)
        x = self.norm(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, depth, head_num, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiheadDiffAttn(embed_dim, depth, head_num)
        self.cross_attn = MultiheadDiffAttn(embed_dim, depth, head_num)    
        self.norm1 = nn.LayerNorm([embed_dim])
        self.norm2 = nn.LayerNorm([embed_dim])
        self.norm3 = nn.LayerNorm([embed_dim])
        self.ff = SwiGLU(embed_dim)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.dropout3 = nn.Dropout(dropout, inplace=True)

    def forward(self, x, y, cross_mask=None):
        skip = x
        x = self.self_attn(x)
        x = self.dropout1(x)
        x = x + skip
        x = self.norm1(x)
        _x = x
        x = self.cross_attn(x, y, attn_mask=cross_mask)
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
        self.embed = nn.ModuleList([nn.Embedding(m, embed_dim) for m in modulo_list])
        self.pe = PositionalEncoding(embed_dim, max_len=max_seq_len)
        self.norm = nn.LayerNorm([embed_dim])
        self.blocks = nn.ModuleList([DecoderBlock(embed_dim, d, head_num) for d in range(block_num)])
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.out_layers = nn.ModuleList([nn.Linear(embed_dim, m) for m in modulo_list])

    def forward(self, x, y, cross_mask=None, offset=0):
        x1 = [x % m for m in modulo_list]
        x = None
        for x2, layer in zip(x1, self.embed):
            if x is None:
                x = layer(x2)
            else:
                x += layer(x2)
        x = self.pe(x, offset=offset)
        x = self.norm(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, y, cross_mask=cross_mask)
        return [layer(x) for layer in self.out_layers]

class Transformer(nn.Module):
    def __init__(self, enc_input_dim, embed_dim, head_num, enc_block_num = 6, dec_block_num = 6, max_enc_seq_len = 5000, max_dec_seq_len = 5000):
        super().__init__()
        self.head_num = head_num
        self.max_len = min(max_enc_seq_len, max_dec_seq_len)
        self.encoder = Encoder(input_dim=enc_input_dim, embed_dim=embed_dim, head_num=head_num, max_seq_len=max_enc_seq_len, block_num=enc_block_num)
        self.decoder = Decoder(embed_dim=embed_dim, head_num=head_num, max_seq_len=max_dec_seq_len, block_num=dec_block_num)
    
    def forward(self, enc_input, dec_input):
        encmask = torch.where(torch.any(enc_input != 0, dim=-1)[:,None,None,:], 0., -float("inf"))
        offset = torch.randint(0, self.max_len - enc_input.shape[1], (1,), device=enc_input.device)
        enc_output = self.encoder(enc_input, attn_mask=encmask, offset=offset)
        output = self.decoder(dec_input, enc_output, cross_mask=encmask, offset=offset)
        return output

@dataclass
class ModelDimensions:
    enc_input_dim: int = encoder_dim
    embed_dim: int = 512
    head_num: int = 16
    enc_block_num: int = 4
    dec_block_num: int = 4
    max_enc_seq_len: int = 256
    max_dec_seq_len: int = 256

class TransformerPredictor(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.head_num = encoder.head_num
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_input):
        encmask = torch.where(torch.any(enc_input != 0, dim=-1)[:,None,None,:], 0., -float("inf"))
        enc_output = self.encoder(enc_input, attn_mask=encmask, offset=0)
        decoder_output = torch.zeros((enc_input.shape[0],max_decoderlen), dtype=torch.long, device=enc_input.device)
        decoder_output[:,0] = decoder_SOT
        for i in range(max_decoderlen):
            outputs = self.decoder(decoder_output, enc_output, cross_mask=encmask, offset=0)
            pred_ids = []
            for decoder_id1 in outputs:
                pred_id1 = torch.argmax(decoder_id1, dim=-1)[:,i]
                pred_ids.append(pred_id1)
            ids = []
            for args in zip(*pred_ids):
                id = calc_predid(*args)
                ids.append(id)
            ids = torch.tensor(ids, dtype=torch.long, device=enc_input.device)
            if torch.all(ids == decoder_EOT):
                break
            if i+1 < max_decoderlen:
                decoder_output[:,i+1] = ids
        return decoder_output[:,1:]

class TransformerEncoderPredictor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.head_num = encoder.head_num
        self.encoder = encoder

    def forward(self, enc_input, encmask):
        enc_output = self.encoder(enc_input, attn_mask=encmask, offset=0)
        return enc_output

class TransformerDecoderPredictor(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.head_num = decoder.head_num
        self.decoder = decoder

    def forward(self, enc_output, decoder_input, crossmask):
        outputs = self.decoder(decoder_input, enc_output, cross_mask=crossmask, offset=0)
        # return [torch.softmax(x, dim=-1) for x in outputs]
        return outputs

if __name__ == '__main__':
    model = Transformer(enc_input_dim=100, embed_dim=512, head_num=8)
    print(model)
    out = model(torch.ones(3,1,100),torch.ones(3,2, dtype=torch.long))
    print(out)
    print([o.shape for o in out])

    # model2 = TransformerPredictor(model.encoder, model.decoder)
    # print(model2)
    # d = model2(torch.ones(3,1,100))
    # print(d)
