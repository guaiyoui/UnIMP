import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import get_activation

class LinearHead(torch.nn.Module):
    def __init__(self, 
                 input_dims, output_dim,
                 hidden_layer_sizes=64,
                 hidden_activation='relu',
                 output_activation=None,
                 dropout=0.):
        super(LinearHead, self).__init__()

        layers = nn.ModuleList()

        input_dim = np.sum(input_dims)

        layer = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_sizes),
            get_activation(hidden_activation),
            nn.Dropout(dropout),
        )
        layers.append(layer)
        input_dim = hidden_layer_sizes

        layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            get_activation(output_activation),
        )
        layers.append(layer)
        self.layers = layers

    def forward(self, inputs, token_emb):
        if torch.is_tensor(inputs):
            inputs = [inputs]
        input_var = torch.cat(inputs, -1)
        for layer in self.layers:
            input_var = layer(input_var)
        return input_var

class SelfAttention(torch.nn.Module):
    '''This class implements the Multi-Head attention.

    Args:
        hid_dim: A integer indicating the hidden dimension.
        n_heads: A integer indicating the number of self attention heads.
        dropout: A float indicating the amount of dropout.
        device: A device to use.
    '''
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0, "Number of heads must be a factor of model dimension"
        # in paper, hid_dim = 512, n_heads = 8

        # query, key, value weight matrices
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        # linear layer to applied after concating the attention head outputs.
        self.fc = nn.Linear(hid_dim, hid_dim)

        # scale factor to be applied in calculation of self attention.
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]
        hidden_dim = query.shape[2]
        assert self.hid_dim == hidden_dim, "Hidden dimensions must match"

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V => [batch_size, sent_len, hidden_dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # Q, K, V => [batch_size, n_heads, sent_len, hid_dim//n_heads]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(query.device)
        # energy => [batch_size, n_heads, sent_len, sent_len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.do(F.softmax(energy, dim=-1))
        # attention => [batch_size, n_heads, sent_len, sent_len]

        x = torch.matmul(attention, V)
        # x => [batch_size, n_heads, sent_len, hid_dim // n_heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x => [batch_size, sent_len, n_heads, hid_dim // n_heads]

        # combine all heads
        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc(x)
        # x => [batch_size, sent_len, hid_dim]
        return x

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, token_emb, gnn_emb):
        # token_emb: [100, 103, 1024]
        # gnn_emb: [100, 1024]
        
        q = self.query(gnn_emb).unsqueeze(1)  # [100, 1, 1024]
        k = self.key(token_emb)  # [100, 103, 1024]
        v = self.value(token_emb)  # [100, 103, 1024]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        
        context = torch.matmul(attn, v)  # [100, 1, 1024]
        
        return token_emb + context.expand(-1, token_emb.size(1), -1)

class WeightedSumFusion(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.weight_gen = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_emb, gnn_emb):
        weights = torch.sigmoid(self.weight_gen(gnn_emb)).unsqueeze(1)  # [100, 1, 1024]
        gnn_context = (token_emb * weights).sum(dim=1, keepdim=True)  # [100, 1, 1024]
        return token_emb + gnn_context.expand(-1, token_emb.size(1), -1)

class GatedFusion(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.gate = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, token_emb, gnn_emb):
        gnn_expanded = gnn_emb.unsqueeze(1).expand(-1, token_emb.size(1), -1)
        gate = torch.sigmoid(self.gate(torch.cat([token_emb, gnn_expanded], dim=-1)))
        return gate * token_emb + (1 - gate) * gnn_expanded

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=8)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8)

    def forward(self, token_emb, gnn_emb):
        token_emb = token_emb.transpose(0, 1)  # [103, 100, 1024]
        gnn_emb = gnn_emb.unsqueeze(0)  # [1, 100, 1024]
        
        token_emb, _ = self.self_attn(token_emb, token_emb, token_emb)
        output, _ = self.cross_attn(token_emb, gnn_emb, gnn_emb)
        
        return output.transpose(0, 1)  # [100, 103, 1024]

class FiLMFusion(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.film_generator = nn.Linear(embed_dim, embed_dim * 2)

    def forward(self, token_emb, gnn_emb):
        film_params = self.film_generator(gnn_emb).unsqueeze(1)  # [100, 1, 2048]
        gamma, beta = film_params.chunk(2, dim=-1)  # Each [100, 1, 1024]
        return gamma * token_emb + beta

class LLMHead(torch.nn.Module):
    def __init__(self, 
                 input_dims, output_dim,
                 hidden_layer_sizes,
                 hidden_activation='relu',
                 output_activation=None,
                 dropout=0.2,
                 relation_type="attention"):
        super(LLMHead, self).__init__()
        # self.fuse_model = SelfAttention(hidden_layer_sizes, n_heads=8, dropout=dropout)
        self.lm_head = nn.Linear(hidden_layer_sizes, output_dim)
        # for param in self.lm_head.parameters():
        #     param.requires_grad = False

        self.lin_gnn = nn.Linear(input_dims, hidden_layer_sizes)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_layer_sizes)
        self.relation_type = relation_type
        if relation_type == "attention":
            self.fuse_model = AttentionFusion(hidden_layer_sizes)
        elif relation_type == "weightedsum":
            self.fuse_model = WeightedSumFusion(hidden_layer_sizes)
        elif relation_type == "gated":
            self.fuse_model = GatedFusion(hidden_layer_sizes)
        elif relation_type == "cross_attn": 
            self.fuse_model = CrossAttentionFusion(hidden_layer_sizes)
        elif relation_type == "film":
            self.fuse_model = FiLMFusion(hidden_layer_sizes)
        else:
            raise ValueError("Not supported relation_type type")
            
        self.ffn = nn.Sequential(
            nn.Linear(hidden_layer_sizes, int(hidden_layer_sizes / 8)),
            nn.ReLU(),
            nn.Linear(int(hidden_layer_sizes / 8), hidden_layer_sizes)
        )

    def forward(self, emb, token_emb):
        # gnn_var = torch.cat([row_emb, col_emb],-1)
        if torch.is_tensor(emb):
            emb = [emb]
        
        gnn_var = torch.cat(emb, -1)
        gnn_var = self.lin_gnn(gnn_var)
        token_emb = token_emb.to(gnn_var.device)
        
        # out = self.fuse_model(token_emb, gnn_var, gnn_var)
        out = self.fuse_model(token_emb, gnn_var)

        # out = self.ln(token_emb + self.dropout(out))
        # out = token_emb + self.dropout(out)
        # out = self.ln(out + self.ffn(out))
        out = self.ln(token_emb + self.ffn(out))
        # out = token_emb + self.ffn(out)

        out = self.lm_head(out)

        return out
        # return self.lm_head(token_emb)
