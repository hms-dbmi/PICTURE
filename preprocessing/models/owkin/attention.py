import torch
import torch.nn as nn

class PatchMultiHeadAttention(nn.Module):
    """
    Multi-head attention for patch embeddings.
    """
    def __init__(self, embed_dim, heads):
        super(PatchMultiHeadAttention, self).__init__()
        self.embed_size = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        assert (
            self.head_dim * heads == embed_dim
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_dim)

    def forward(self, values, keys, query):
        N = query.shape[0]

        # Split the embeddings into 'self.heads' pieces
        values = self.values(values).view(N, -1, self.heads, self.head_dim)
        keys = self.keys(keys).view(N, -1, self.heads, self.head_dim)
        queries = self.queries(query).view(N, -1, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, -1, self.heads * self.head_dim
        )
        return self.fc_out(out)

    def maxpool(self, attention_output):
        return attention_output.max(dim=0)[0]


if __name__ == "__main__":
    EMBEDDING_DIM = 768
    NUM_WINDOWS = 10
    NUM_PATCHES_PER_WINDOW = 50
    patch_embeddings = torch.randn(NUM_PATCHES_PER_WINDOW, EMBEDDING_DIM)
    window_embeddings = torch.randn(NUM_WINDOWS, EMBEDDING_DIM)

    patch_attention = PatchMultiHeadAttention(EMBEDDING_DIM, heads=8)
    window_attention = PatchMultiHeadAttention(EMBEDDING_DIM, heads=8)

    patch_attn_output = patch_attention(
        patch_embeddings, patch_embeddings, patch_embeddings
    )
    window_attn_output = window_attention(
        window_embeddings, window_embeddings, window_embeddings
    )

    # [1] returns indices of max values
    fused_window_embedding = patch_attn_output.max(dim=0)[0]
    fused_bag_embedding = window_attn_output.max(dim=0)[0]

    print(fused_window_embedding.shape, fused_bag_embedding.shape)

class SimpleAttention(nn.Module):
    def __init__(self, embed_dim: int, non_linearity: nn.Module = nn.Tanh()):
        super(SimpleAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            non_linearity,
            nn.Linear(embed_dim, 1),
        )


    def forward(self, embeddings):
        attention_weights = torch.softmax(self.attention(embeddings), dim=0)
        weighted_embeddings = torch.mul(attention_weights, embeddings)

        return {
            "weighted_embeddings": weighted_embeddings,
            "attention_weights": attention_weights,
        }


class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SingleHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        # print("q.shape", q.shape)
        # print("k.shape", k.shape)
        # print("v.shape", v.shape)

        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_scores = attention_scores / (self.embed_dim ** 0.5)
        attention_weights = self.softmax(attention_scores)
        # print("attention_weights.shape", attention_weights.shape)

        attended_values = torch.matmul(attention_weights, v)
        return attended_values
