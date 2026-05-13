import sys
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class caps_att(nn.Module):
    """
    Args:
    """
    def __init__(self, num_iterations, num_capsules, dim, out_dim, num_heads=8):
        super(caps_att, self).__init__()
        self.dp = nn.Dropout(0.1)
        self.num_capsules = num_capsules
        self.num_iterations = num_iterations
        self.out_layer = nn.Linear(dim, out_dim)
        self.att_size = dim
        self.W_h = nn.Parameter(torch.Tensor(dim, self.att_size))
        nn.init.xavier_uniform_(self.W_h)
        self.W_f = nn.Parameter(torch.Tensor(dim, self.att_size))
        nn.init.xavier_uniform_(self.W_f)
        self.layer_norm = nn.LayerNorm(dim)
        self.activation = nn.SELU()
        self.group_transforms = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(4)
        ])
        for transform in self.group_transforms:
            nn.init.orthogonal_(transform.weight)

    def forward(self, query, feat, feat_mask=None):

        query = query @ self.W_h
        feat = feat @ self.W_f

        feat_mask = feat_mask.squeeze(1).squeeze(1)  # b, feat_len

        b = torch.zeros(feat.shape[0], feat.shape[1], device=feat.device)
        b = b.masked_fill(feat_mask, -1e18)
        for i in range(self.num_iterations):
            c = F.softmax(b, dim=1)  # b,feat_len

            transformed_feats = []
            for transform in self.group_transforms:
                transformed_feat = transform(feat)  # b, feat_len, dim
                transformed_feats.append(transformed_feat)
            transformed_feats = torch.stack(transformed_feats, dim=0)  # [num_transforms, b, feat_len, dim]
            feat_group = transformed_feats.mean(dim=0)  # [b, feat_len, dim]

            if i == self.num_iterations - 1:
                outputs = (c.unsqueeze(-1) * feat_group).sum(dim=1)  # b,dim
                outputs = self.activation(outputs)
                query = query + outputs  # b,dim
            else:
                delta_b = (query.unsqueeze(1) * feat_group).sum(dim=-1)  # b,feat_len
                delta_b = (delta_b - delta_b.mean(dim=1, keepdim=True)) / (delta_b.std(dim=1, keepdim=True) + 1e-9)
                b = b + delta_b
                outputs = (c.unsqueeze(-1) * feat_group).mean(dim=1)  # b,dim
                outputs = self.activation(outputs)
                query = query + outputs  # b,dim
            query = self.layer_norm(query)

        return self.out_layer(query)
