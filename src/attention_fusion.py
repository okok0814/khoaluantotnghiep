import torch
from torch import nn
import torch.nn.functional as F


class AttentionGatedFusion(nn.Module):
    def __init__(self, embedding_dim=512, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Sigmoid(),
        )
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, image_embedding, text_embedding, return_gate=False):
        image_embedding = F.normalize(image_embedding, dim=-1)
        text_embedding = F.normalize(text_embedding, dim=-1)

        joint = torch.cat([image_embedding, text_embedding], dim=-1)
        gate = self.gate_network(joint)

        fused = gate * image_embedding + (1.0 - gate) * text_embedding
        query_embedding = fused + self.projection(fused)
        query_embedding = F.normalize(query_embedding, dim=-1)

        if return_gate:
            return query_embedding, gate
        return query_embedding


def in_batch_contrastive_loss(query_embedding, target_embedding, temperature=0.07):
    query_embedding = F.normalize(query_embedding, dim=-1)
    target_embedding = F.normalize(target_embedding, dim=-1)

    logits = (query_embedding @ target_embedding.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss, logits