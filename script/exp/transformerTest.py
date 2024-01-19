from torch import nn
import torch
import numpy as np
import copy
import math

from torch.autograd import Variable


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).cuda()
        return x


class MultiHead(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate) -> None:
        super().__init__()
        self.head_dim = model_dim // n_head
        self.n_head = n_head
        self.model_dim = model_dim
        self.wq = nn.Linear(model_dim, n_head * self.head_dim)
        self.wk = nn.Linear(model_dim, n_head * self.head_dim)
        self.wv = nn.Linear(model_dim, n_head * self.head_dim)

        self.o_dense = nn.Linear(model_dim, model_dim)
        self.o_drop = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.attention = None

    def forward(self, q, k, v, mask, training):
        # 残差
        residule = q

        # 线性传播
        key = self.wk(k)  # [n, step, n_head * head_dim] 分别为[样本数，用多少个连续样本预测一个输出，特征数]
        query = self.wq(q)
        value = self.wv(v)

        # 参数切割
        key = self.split_heads(key)
        query = self.split_heads(query)
        value = self.split_heads(value)
        context = self.scaled_dot_product_attention(query, key, value, mask)
        o = self.o_dense(context) # [n, step, model_dim]
        o = self.o_drop(o)

        o = self.layer_norm(residule + o)
        return o


    def split_heads(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1], self.n_head, self.head_dim))
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dk = torch.tensor(k.shape[-1]).type(torch.float)
        score = torch.matmul(q, k.permute(0, 1, 3, 2)) / (torch.sqrt(dk) + 1e-8) # [n, n_head, step, step]
        if mask is not None:
            score = score.masked_fill(mask, -np.inf)
        self.attention = torch.softmax(score, dim=-1)
        context = torch.matmul(self.attention, v)   # [n, num_head, step, head_dim]
        context = context.permute(0, 2, 1, 3)       # [n, step, num_head, head_dim]
        context = context.reshape((context.shape[0], context.shape[1], -1))
        return context  # [n, step, model_dim]


class Test(nn.Module):

    def __init__(self, n_heads, model_dim, drop_rate) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.model_dim = model_dim
        self.head_dim = model_dim // n_heads

        self.wq = nn.Linear(model_dim, n_heads * self.head_dim)
        self.wk = nn.Linear(model_dim, n_heads * self.head_dim)
        self.wv = nn.Linear(model_dim, n_heads * self.head_dim)

        self.o_dense = nn.Linear(self.model_dim, self.model_dim)
        self.o_drop = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(model_dim)

        self.drop_rate = drop_rate
        self.attention = None

    def forward(self, q, k, v, mask, training):

        residule = q

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        query = self.split_to_head(q)
        key = self.split_to_head(k)
        value = self.split_to_head(v)

        context = self.scaled_up_dot_attention(query, key, value, mask)

        o = self.o_dense(context)
        o = self.o_drop(o)
        o = self.layer_norm(residule + o)

        return o

    def split_to_head(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1], self.n_heads, self.head_dim))
        return x.permute(0, 2, 1, 3)

    def scaled_up_dot_attention(self, query, key, value, mask):
        dk = torch.tensor(key.shape[-1]).type(torch.float16)
        score = torch.matmul(query, key.permute(0, 1, 3, 2)) / (torch.sqrt(dk) + 1e-8)
        if mask:
            self.attention = score.masked_fill(mask, -np.inf)
        context = torch.matmul(self.attention, value)
        context = context.permute(0, 2, 1, 3)
        context = torch.reshape(context, (context.shape[0], context.shape[1], -1))
        return context

class PositionWiseFFN(nn.Module):
    def __init__(self, model_dim, dropout=0.0) -> None:
        super().__init__()
        dff = model_dim * 4
        self.l = nn.Linear(model_dim, dff)
        self.o = nn.Linear(dff, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def foward(self, x):
        o = torch.relu(self.l(x))
        o = self.o(o)
        o = self.dropout(o)

        o = self.layer_norm(x + o)
        return o

class PositionWiseFFNTest(nn.Module):

    def __init__(self, model_dim, drop_rate) -> None:
        super().__init__()

        dff = model_dim * 4
        self.l = nn.Linear(model_dim, dff)
        self.o = nn.Linear(dff, model_dim)
        self.o_drop = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):

        l = torch.relu(self.l(x))
        o = self.o(l)
        o = self.o_drop(o)
        o = self.layer_norm(x + o)
        return o


class EncoderLayer(nn.Module):
    def __init__(self, n_head, emb_dim, drop_rate):
        super().__init__()
        self.mh = MultiHead(n_head, emb_dim, drop_rate)
        self.ffh = PositionWiseFFN(emb_dim, drop_rate)

    def forward(self, xz, training, mask):
        context = self.mh(xz, xz, xz, mask, training)
        o = self.ffh(context)
        return o

    def get_clones(module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DecoderLayer(nn.Module):
    def __init__(self, n_head, emb_dim, drop_rate):
        super().__init__()
        self.mh = nn.ModuleList([MultiHead(n_head, emb_dim, drop_rate) for _ in range(2)])
        self.ffn = PositionWiseFFN(emb_dim, drop_rate)

    def forward(self, yz, xz, training, yz_look_ahead_mask, xz_pad_mask):
        dec_output = self.mh[0](yz, yz, yz, yz_look_ahead_mask, training)   # [n, step, model_dim]
        dec_output = self.mh[1](dec_output, xz, xz, xz_pad_mask, training)  # [n, step, model_dim]
        dec_output = self.ffn(dec_output)

        return dec_output


class Encoder(nn.Module):
    def __init__(self, vocab_size, model_dim, N, heads, drop_rate):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.pe = PositionalEncoder(model_dim)
        self.layers = EncoderLayer.get_clones(EncoderLayer(model_dim, heads, drop_rate), N)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, drop_rate):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = DecoderLayer.get_clones(DecoderLayer(d_model, heads, drop_rate), N)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return x

    def get_clones(module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, drop_rate):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, drop_rate)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, drop_rate)
        self.out = nn.Linear(d_model, trg_vocab)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        output = self.softmax(output)
        return output