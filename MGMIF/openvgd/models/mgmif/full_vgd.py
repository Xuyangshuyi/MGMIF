import torch
import torch.nn as nn
import torch.nn.functional as F
from openvgd.models.mcan.modules import AttFlat, LayerNorm, MCA_ED
from transformers import BertModel
from transformers import AlbertModel
from transformers import RobertaModel

class Net_Full(nn.Module):
    def __init__(self, __C, init_dict):
        super(Net_Full, self).__init__()
        self.__C = __C

        if self.__C.BERT_ENCODER:
            # self.encoder = AlbertModel.from_pretrained(__C.PRETRAINED_PATH)
            self.bert_encoder = RobertaModel.from_pretrained(__C.PRETRAINED_PATH)

        elif not self.__C.BERT_ENCODER and self.__C.USE_BERT:
            # self.bert_layer = AlbertModel.from_pretrained(__C.PRETRAINED_PATH, output_hidden_states=True)
            self.bert_layer = BertModel.from_pretrained(__C.PRETRAINED_PATH, output_hidden_states=True)
            #self.bert_layer = RobertaModel.from_pretrained(__C.PRETRAINED_PATH, output_hidden_states=True)
            # Freeze BERT layers
            for param in self.bert_layer.parameters():
                param.requires_grad = False

        elif self.__C.GLOVE_FEATURE:
            self.embedding = nn.Embedding(num_embeddings=init_dict['token_size'], embedding_dim=__C.WORD_EMBED_SIZE)
            self.embedding.weight.data.copy_(torch.from_numpy(init_dict['pretrained_emb']))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HSIZE,
            num_layers=1,
            batch_first=True
        )

        imgfeat_linear_size = __C.FRCNFEAT_SIZE
        if __C.BBOX_FEATURE:
            self.bboxfeat_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.imgfeat_linear = nn.Linear(imgfeat_linear_size, __C.HSIZE)

        self.backnone = MCA_ED(__C)
        # self.backnone = GED(__C)
        self.attflat_x = AttFlat(__C)
        self.attfc_y = nn.Linear(__C.HSIZE, __C.ATTFLAT_OUT_SIZE)
        self.bert_conn = nn.Linear(__C.WORD_EMBED_SIZE, __C.HSIZE)
        self.proj_norm = LayerNorm(__C.ATTFLAT_OUT_SIZE)
        self.proj_scores = nn.Linear(__C.ATTFLAT_OUT_SIZE, 1)
        self.proj_reg = nn.Linear(__C.ATTFLAT_OUT_SIZE, 4)
        self.linear_layer = torch.nn.Linear(2048, 1024)

    def forward(self, input):
        frcn_feat, bbox_feat, ques_ix = input

        # with torch.no_grad():
        # Make mask for attention learning
        if self.__C.USE_BERT:
            x_mask = self.make_mask(ques_ix[:, 1:-1].unsqueeze(2))
        else:
            x_mask = self.make_mask(ques_ix.unsqueeze(2))
        y_mask = self.make_mask(frcn_feat)

        # Pre-process Language Feature
        if self.__C.BERT_ENCODER:
            outputs = self.bert_encoder(ques_ix)
            last_hidden_state = outputs[0]
            x_in = last_hidden_state[:, 1:-1, :]  # remove CLS and SEP, making this to max_token=14 # (64,14,768)

        elif not self.__C.BERT_ENCODER and self.__C.USE_BERT:
            outputs = self.bert_layer(ques_ix)
            hidden_states = outputs[2]  # All Outputs of bert model's hidden
            concat_layers = torch.cat([hidden_states[i] for i in [-1, -2, -3, -4, -5]], dim=-1)
            # print(concat_layers.shape)
            concat_layers = concat_layers[:, 1:-1, :]
            # print(concat_layers.shape)
            if self.__C.USE_LSTM:
                x_in, _ = self.lstm(concat_layers)  # into LSTM
            else:
                x_in = self.bert_conn(concat_layers)  # into BERT_CONN_Layer

        elif self.__C.GLOVE_FEATURE:
            lang_feat = self.embedding(ques_ix) # (64, 15, 300)
            x_in, _ = self.lstm(lang_feat) # (64, 15, 512)


        # Pre-process Image Feature
        if self.__C.BBOX_FEATURE:
            bbox_feat = self.bboxfeat_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        y_in = self.imgfeat_linear(frcn_feat)

        # print(x_in.shape, y_in.shape)

        # xy_out = self.backnone(x_in, y_in, x_mask, y_mask)  # (64, 15, 512) (64, 100, 512)
        x_out, y_out = self.backnone(x_in, y_in, x_mask, y_mask) # (64, 15, 512) (64, 100, 512)
        x_out = self.attflat_x(x_out, x_mask).unsqueeze(1)  # (64, 1, 1024)
        # x_out = self.attfc_y(x_out)
        y_out = self.attfc_y(y_out)  # (64, 100, 1024)
        xy_out = x_out + y_out

        xy_out = self.proj_norm(xy_out)
        pred_scores = self.proj_scores(xy_out).squeeze(-1)
        if self.__C.SCORES_LOSS == 'kld':
            pred_scores = F.log_softmax(pred_scores, dim=-1)
        pred_reg = self.proj_reg(xy_out)
        return pred_scores, pred_reg

    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature), dim=-1) == 0).unsqueeze(1).unsqueeze(2)
