import torch
import torch.nn as nn

class AttentionHead(nn.Module):
    def __init__(self, emb_size, head_size):
        super().__init__()
        self.qkv = nn.Linear(emb_size, 3 * head_size, bias=True)
        self.out = nn.Linear(head_size, emb_size)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        attn_weights = (q @ k.transpose(-2, -1)) / k.size(-1)**0.5
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = attn_weights @ v
        return self.out(attn_output)

class FeedForward(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.ReLU(),
            nn.Linear(4 * emb_size, emb_size)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, head_size):
        super().__init__()
        self.attn = AttentionHead(emb_size, head_size)
        self.norm1 = nn.LayerNorm(emb_size)
        self.ff = FeedForward(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size=256, num_heads=4, num_layers=2, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size)
        self.pos = nn.Parameter(torch.randn(1, 128, emb_size))
        self.layers = nn.Sequential(*[
            TransformerBlock(emb_size, emb_size // num_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(emb_size, num_classes)

    def forward(self, x, mask=None):
        x = self.embed(x) + self.pos[:, :x.size(1)]
        x = self.layers(x)
        x = x.mean(dim=1)
        return self.fc(x)