
# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.mgmif.caps import caps_att
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class AttFlat_caps(nn.Module):
    def __init__(self, __C):
        super(AttFlat_caps, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.HIDDEN_SIZE
        )
    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.update_layer = nn.Linear(2 * __C.HIDDEN_SIZE, __C.HIDDEN_SIZE, bias=False)
        self.gate = nn.Linear(2 * __C.HIDDEN_SIZE, __C.HIDDEN_SIZE, bias=False)

    def forward(self, x, x_mask):
        x_1 = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x_2 = self.norm2(x_1 + self.dropout2(
            self.ffn(x_1)
        ))

        inputs = torch.cat([x_2, x_1], dim=-1)

        f_t = torch.tanh(self.update_layer(inputs))

        g_t = torch.sigmoid(self.gate(inputs))

        x_feat = g_t * f_t + (1 - g_t) * x_2

        return x_feat


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, x, y_mask, x_mask):
        # image self-attention
        y = self.norm1(y + self.dropout1(
            self.mhatt1(v=y, k=y, q=y, mask=y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.mhatt2(v=x, k=x, q=y, mask=x_mask)
        ))

        y = self.norm3(y + self.dropout3(
            self.ffn(y)
        ))

        return y


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list_1 = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])
        self.dec_list_2 = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

        self.attflat_lang = AttFlat_caps(__C)
        self.attflat_img = AttFlat_caps(__C)

        self.caps_feat_img = caps_att(num_iterations=3, num_capsules=150, dim=__C.HIDDEN_SIZE,
                                             out_dim=__C.HIDDEN_SIZE)
        self.caps_feat_lang = caps_att(num_iterations=3, num_capsules=150, dim=__C.HIDDEN_SIZE,
                                              out_dim=__C.HIDDEN_SIZE)
        self.linear_img = nn.Linear(__C.HIDDEN_SIZE * 2, __C.HIDDEN_SIZE * 2)
        self.linear_lang = nn.Linear(__C.HIDDEN_SIZE * 2, __C.HIDDEN_SIZE * 2)

    def forward(self, x, y, x_mask, y_mask):
        for enc in self.enc_list:
            x = enc(x, x_mask)
        lang_query_reserve = self.attflat_lang(x, x_mask)
        lang_query_branch1 = lang_query_branch2 = lang_query_reserve
        y_branch1 = y_branch2 = y

        for dec in self.dec_list_1:
            y_branch1 = dec(y_branch1, x, y_mask, x_mask)
        img_query_branch1 = self.caps_feat_img(lang_query_branch1, y_branch1, y_mask)  # b,1024 b,100
        lang_query_branch1 = self.caps_feat_lang(img_query_branch1, x, x_mask)  # b,1024 b,14

        img_feat_query = self.attflat_img(y_branch1, y_mask)  # torch.Size([64, 512])
        img_feat_f = torch.cat([img_feat_query, img_query_branch1], dim=-1)  # b,2048
        lang_feat_f = torch.cat([lang_query_reserve, lang_query_branch1], dim=-1)  # b,2048

        gate = F.sigmoid(self.linear_lang(lang_feat_f) + self.linear_img(img_feat_f))

        proj_feat = gate * lang_feat_f + (1-gate) * img_feat_f

        mask = adaptive_mask(y_branch2)

        y_branch2 = y_branch2 + (y_branch2 * mask.float())

        for dec in self.dec_list_2:
            y_branch2 = dec(y_branch2, x, y_mask, x_mask)
        img_query_branch2 = self.caps_feat_img(lang_query_branch2, y_branch2, y_mask)  # b,1024 b,100
        lang_query_branch2 = self.caps_feat_lang(img_query_branch2, x, x_mask)  # b,1024 b,14

        img_feat_query_2 = self.attflat_img(y_branch2, y_mask)  # torch.Size([64, 512])
        img_feat_f_2 = torch.cat([img_feat_query_2, img_query_branch2], dim=-1)  # b,2048
        lang_feat_f_2 = torch.cat([lang_query_reserve, lang_query_branch2], dim=-1)  # b,2048

        gate = F.sigmoid(self.linear_lang(lang_feat_f_2) + self.linear_img(img_feat_f_2))

        proj_feat_2 = gate * lang_feat_f_2 + (1 - gate) * img_feat_f_2

        return proj_feat, proj_feat_2

def adaptive_mask(Y):
    sorted_Y, _ = Y.sort(dim=1, descending=True)

    differences = sorted_Y[:, :-1] - sorted_Y[:, 1:]

    _, max_diff_indices = differences.max(dim=1)

    threshold_values = sorted_Y.gather(1, max_diff_indices.unsqueeze(1))

    mask = Y > threshold_values

    return mask

def att_mask(att, att_mask):
    value, att_argmax = att.topk(att.size(1), dim=1, largest=True)
    b = att.size(1) - att_mask.sum(dim=1)
    b = b * 0.5
    mid_list = []
    for i in range(att.size(0)):
        mid = value[i][int(b[i].item())]
        mid_list.append(mid)
    mid_t = torch.stack(mid_list, dim=0).unsqueeze(-1)  # b,1
    mid_t = mid_t.repeat(1, att.size(1))
    mask = mid_t > att
    return mask

