import torch
from torch import nn
from dataclasses import dataclass

from util_func import modulo_list, calc_predid, feature_dim
from const import decoder_SOT, decoder_EOT, max_decoderlen, encoder_add_dim

encoder_dim = feature_dim + encoder_add_dim

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len = 5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, dim))

    def forward(self, x, offset = 0):
        pe = self.pe[offset:offset+x.shape[0]].unsqueeze(1)
        return x + pe

class FeedForward(nn.Module):
    def __init__(self, dim, dropout = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GLU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.in_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def calc_qkv(self, query, key, value, k_cache, v_cache):
        return self._in_projection_packed(query, key, value)

    def forward(self, query, key=None, value=None, attn_mask=None, is_causal=False, k_cache = None, v_cache = None):
        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape

        head_dim = embed_dim // self.num_heads
        q, k, v = self.calc_qkv(query, key, value, k_cache, v_cache)

        #
        # reshape q, k, v for multihead attention and make them batch first
        #
        q = q.view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        # adjust dropout probability
        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout

        q = q.view(bsz, self.num_heads, tgt_len, head_dim)
        k = k.view(bsz, self.num_heads, src_len, head_dim)
        v = v.view(bsz, self.num_heads, src_len, head_dim)

        attn_output = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, is_causal
        )
        attn_output = (
            attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        )

        attn_output = nn.functional.linear(attn_output, self.out_proj.weight, None)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        return attn_output

    def _in_projection_packed(self, query, key, value):
        E = query.size(-1)
        if key is value:
            if query is key:
                # self-attention
                proj = nn.functional.linear(query, self.in_proj.weight, None)
                # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
                proj = (
                    proj.unflatten(-1, (3, E))
                    .unsqueeze(0)
                    .transpose(0, -2)
                    .squeeze(-2)
                    .contiguous()
                )
                return proj[0], proj[1], proj[2]
            else:
                # encoder-decoder attention
                w_q, w_kv = self.in_proj.weight.split([E, E * 2])
                q_proj = nn.functional.linear(query, w_q, None)
                kv_proj = nn.functional.linear(key, w_kv, None)
                # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
                kv_proj = (
                    kv_proj.unflatten(-1, (2, E))
                    .unsqueeze(0)
                    .transpose(0, -2)
                    .squeeze(-2)
                    .contiguous()
                )
                return q_proj, kv_proj[0], kv_proj[1]
        else:
            w_q, w_k, w_v = self.in_proj.weight.chunk(3)
            return nn.functional.linear(query, w_q, None), nn.functional.linear(key, w_k, None), nn.functional.linear(value, w_v, None)

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm([dim])
        self.norm2 = nn.LayerNorm([dim])
        self.ff = FeedForward(dim)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)

    def forward(self, x, attn_mask=None):
        Q = K = V = x
        x = self.mha(Q, K, V, attn_mask=attn_mask)
        x = self.dropout1(x)
        x = x + Q
        x = self.norm1(x)
        _x = x
        x = self.ff(x)
        x = self.dropout2(x)
        x = x + _x + Q
        x = self.norm2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, dim, head_num, max_seq_len=5000, block_num = 6, dropout = 0.1):
        super().__init__()
        self.dim = dim
        self.head_num = head_num
        self.embed = nn.Linear(input_dim, dim)
        self.pe = PositionalEncoding(dim, max_len=max_seq_len)
        self.norm = nn.LayerNorm([dim])
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.blocks = nn.ModuleList([EncoderBlock(dim, head_num) for _ in range(block_num)])        

    def forward(self, x, attn_mask=None, offset=0):
        x = self.embed(x)
        x = self.pe(x, offset=offset)
        x = self.norm(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, dim, head_num, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, head_num)
        self.cross_attn = MultiHeadAttention(dim, head_num)    
        self.norm1 = nn.LayerNorm([dim])
        self.norm2 = nn.LayerNorm([dim])
        self.norm3 = nn.LayerNorm([dim])
        self.ff = FeedForward(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, y, self_mask=None, cross_mask=None):
        Q = K = V = x
        skip = x
        x = self.self_attn(Q, K, V, attn_mask=self_mask)
        x = self.dropout1(x)
        x = x + Q
        x = self.norm1(x)
        Q = x
        K = V = y
        x = self.cross_attn(Q, K, V, attn_mask=cross_mask)
        x = self.dropout2(x)
        x = x + Q
        x = self.norm2(x)
        _x = x
        x = self.ff(x)
        x = self.dropout3(x)
        x = x + _x + skip
        x = self.norm3(x)
        return x

# https://github.com/KindXiaoming/grow-crystals
# Harmonic Loss Trains Interpretable AI Models
# https://arxiv.org/abs/2502.01628
class DistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = eps
        
    def forward(self, x):
        # x: (B, N)
        # w: (V, N)
        # dist_sq: (B, V)
        w = self.weight
        wx = torch.matmul(x, w.mT) # (B, V)
        ww = torch.norm(w, dim=-1)**2 # (V,)
        xx = torch.norm(x, dim=-1)**2 # (B,)

        dist_sq = ww.unsqueeze(-2) + xx.unsqueeze(-1) - 2 * wx + self.eps
        dist_sq = dist_sq / torch.min(dist_sq, dim=-1, keepdim = True)[0]
        prob = (dist_sq)**(-self.n)
        prob = prob/torch.sum(prob, dim=-1, keepdim=True)
        logits = torch.log(prob)
        return logits

class Decoder(nn.Module):
    def __init__(self, dim, head_num, max_seq_len=5000, block_num = 6, dropout = 0.1):
        super().__init__()
        self.head_num = head_num
        self.embed = nn.ModuleList([nn.Embedding(m, dim) for m in modulo_list])
        self.pe = PositionalEncoding(dim, max_len=max_seq_len)
        self.norm = nn.LayerNorm([dim])
        self.blocks = nn.ModuleList([DecoderBlock(dim, head_num) for _ in range(block_num)])
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.out_layers = nn.ModuleList([DistLayer(dim, m) for m in modulo_list])

    def forward(self, x, y, self_mask=None, cross_mask=None, offset=0):
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
            x = block(x, y, self_mask=self_mask, cross_mask=cross_mask)
        return [layer(x) for layer in self.out_layers]

class Transformer(nn.Module):
    def __init__(self, enc_input_dim, dim, head_num, enc_block_num = 6, dec_block_num = 6, max_enc_seq_len = 5000, max_dec_seq_len = 5000):
        super().__init__()
        self.head_num = head_num
        self.encoder = Encoder(input_dim=enc_input_dim, dim=dim, head_num=head_num, max_seq_len=max_enc_seq_len, block_num=enc_block_num)
        self.decoder = Decoder(dim=dim, head_num=head_num, max_seq_len=max_dec_seq_len, block_num=dec_block_num)
    
    def forward(self, enc_input, dec_input):
        encmask = torch.any(enc_input != 0, dim=-1)
        encmask = encmask.transpose(1,0)
        encmask = encmask[:,None,None,:].expand(-1,-1,encmask.shape[-1],-1)
        decmask = torch.ones(dec_input.shape[0], dec_input.shape[0], dtype=torch.bool, device=enc_input.device).tril(diagonal=0).unsqueeze(0).unsqueeze(0)
        offset = torch.randint(0, enc_input.shape[0] // 2, (1,), device=enc_input.device)
        enc_output = self.encoder(enc_input, attn_mask=encmask, offset=offset)
        output = self.decoder(dec_input, enc_output, self_mask=decmask, cross_mask=encmask, offset=offset)
        return output

@dataclass
class ModelDimensions:
    enc_input_dim: int = encoder_dim
    dim: int = 512
    head_num: int = 16
    enc_block_num: int = 8
    dec_block_num: int = 8
    max_enc_seq_len: int = 256
    max_dec_seq_len: int = 256

class TransformerPredictor(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.head_num = encoder.head_num
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_input):
        encmask = torch.any(enc_input != 0, dim=-1)
        encmask = encmask.transpose(1,0)
        encmask = encmask[:,None,None,:].expand(-1,-1,encmask.shape[-1],-1)
        enc_output = self.encoder(enc_input, attn_mask=encmask, offset=0)
        decoder_output = torch.zeros((max_decoderlen, enc_input.shape[1]), dtype=torch.long, device=enc_input.device)
        decmask = torch.ones(max_decoderlen, max_decoderlen, dtype=torch.bool, device=enc_input.device).tril(diagonal=0).unsqueeze(0).unsqueeze(0)
        decoder_output[0,:] = decoder_SOT
        for i in range(max_decoderlen):
            outputs = self.decoder(decoder_output, enc_output, self_mask=decmask, cross_mask=encmask, offset=0)
            pred_ids = []
            for decoder_id1 in outputs:
                pred_id1 = torch.argmax(decoder_id1, dim=-1)[i]
                pred_ids.append(pred_id1)
            ids = []
            for args in zip(*pred_ids):
                id = calc_predid(*args)
                ids.append(id)
            ids = torch.tensor(ids, dtype=torch.long, device=enc_input.device)
            if torch.all(ids == decoder_EOT):
                break
            if i+1 < max_decoderlen:
                decoder_output[i+1,:] = ids
        return decoder_output[1:,:]

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

    def forward(self, enc_output, decoder_input, selfmask, crossmask):
        outputs = self.decoder(decoder_input, enc_output, self_mask=selfmask, cross_mask=crossmask, offset=0)
        return outputs

if __name__ == '__main__':
    model = Transformer(enc_input_dim=100, dim=512, head_num=8)

    model2 = TransformerPredictor(model.encoder, model.decoder)
    print(model2)
    d = model2(torch.ones(3,1,100))
    print(d)
