import numpy as np
import torch, en_vectors_web_lg, spacy, re, json, random, copy, glob
import torch.utils.data as Data
from openvgd.utils.bbox import bbox_overlaps
from openvgd.utils.bbox_transform import bbox_transform
from transformers import BertTokenizer
from transformers import BertTokenizerFast
from transformers import AlbertTokenizer
from transformers import RobertaTokenizer

class DataSet(Data.Dataset):
    def __init__(self, __C, RUN_MODE):
        self.__C = __C
        self.RUN_MODE = RUN_MODE
        # self.tokenizer = AlbertTokenizer.from_pretrained(self.__C.PRETRAINED_PATH)
        # self.tokenizer = BertTokenizer.from_pretrained(self.__C.PRETRAINED_PATH)
        # self.tokenizer = RobertaTokenizer.from_pretrained(self.__C.PRETRAINED_PATH)
        self.tokenizer = None

        # Loading all image paths
        frcn_feat_path_list = []
        if __C.IMGFEAT_MODE in ['coco_mrcn']:
            frcn_feat_path_list += glob.glob(__C.IMGFEAT_PATH[__C.IMGFEAT_MODE][__C.DATASET] + '*.npz')
        else:
            for set_ in __C.IMGFEAT_PATH[__C.IMGFEAT_MODE]:
                frcn_feat_path_list += glob.glob(__C.IMGFEAT_PATH[__C.IMGFEAT_MODE][set_] + '*.npz')

        # Loading question word list
        stat_refs_list = []
        for set_ in __C.REF_PATH[__C.DATASET]:
            stat_refs_list += json.load(open(__C.REF_PATH[__C.DATASET][set_], 'r'))

        self.refs_list = []
        for split_ in __C.SPLIT[RUN_MODE].split('+'):
            self.refs_list += json.load(open(__C.REF_PATH[__C.DATASET][split_], 'r'))

        self.data_size = self.refs_list.__len__()
        print(' ========== Dataset size:', self.data_size)

        # {image id} -> {image feature absolutely path}
        self.iid_to_path = self.img_feat_path_load(frcn_feat_path_list, __C.IMGFEAT_MODE)

        # Tokenize
        # self.token_to_ix, self.pretrained_emb = self.tokenize(stat_refs_list, __C.GLOVE_FEATURE)
        self.token_to_ix, self.pretrained_emb = self.tokenize(stat_refs_list, __C.GLOVE_FEATURE, self.tokenizer)
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Question token vocab size:', self.token_size)


    def img_feat_path_load(self, path_list, IMGFEAT_MODE):
        iid_to_path = {}
        for ix, path in enumerate(path_list):
            if IMGFEAT_MODE in ['coco_mrcn']:
                iid = path.split('/')[-1].split('.')[0]
            else:
                iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
            # print(iid)
            iid_to_path[iid] = path

        return iid_to_path


    def tokenize(self, stat_refs_list, use_glove, tokenizer):
        if tokenizer is not None:
            token_to_ix = {}

            for ref_ in stat_refs_list:
                words = ref_['tokens']

                for tup in words:
                    if tup not in token_to_ix:
                        token_to_ix[tup] = len(token_to_ix)
                pretrained_emb = None
            # print(token_to_ix)
        else:
            token_to_ix = {
                'PAD': 0,
                'UNK': 1,
                'CLS': 2,
            }

            spacy_tool = None
            pretrained_emb = []
            if use_glove:
                spacy_tool = en_vectors_web_lg.load()
                pretrained_emb.append(spacy_tool('PAD').vector)
                pretrained_emb.append(spacy_tool('UNK').vector)
                pretrained_emb.append(spacy_tool('CLS').vector)

            for ref_ in stat_refs_list:
                words = ref_['tokens']
                # print(words)
                for word in words:
                    if word not in token_to_ix:
                        token_to_ix[word] = len(token_to_ix)
                        if use_glove:
                            pretrained_emb.append(spacy_tool(word).vector)
            pretrained_emb = np.array(pretrained_emb)
            # print(token_to_ix)

        return token_to_ix, pretrained_emb


    def __getitem__(self, idx):
        refs = self.refs_list[idx]
        max_token = 16 if self.__C.USE_BERT else 14
        query_ix_iter = self.proc_query(refs, max_token, self.tokenizer)

        frcn_feat = np.load(self.iid_to_path[str(refs['image_id'])])
        if self.__C.IMGFEAT_MODE in ['coco_mrcn']:
            # print(frcn_feat['fc7'].shape)  # 100, 2048
            # print(frcn_feat['pool5'].shape)  # 100, 1024
            frcn_feat_x = np.concatenate((frcn_feat['fc7'], frcn_feat['pool5']), axis=-1)
        else:
            frcn_feat_x = frcn_feat['x'].transpose((1, 0))
        frcn_feat_iter = self.proc_img_feat(frcn_feat_x, img_feat_pad_size=100)
        bbox_feat_iter = self.proc_img_feat(
            self.proc_bbox_feat(
                frcn_feat['bbox'],
                (frcn_feat['image_h'], frcn_feat['image_w'])
            ),
            img_feat_pad_size=100
        )
        bbox_iter = self.proc_img_feat(frcn_feat['bbox'], img_feat_pad_size=100)
        img_shape_iter = np.array([frcn_feat['image_h'], frcn_feat['image_w']])

        gt_bbox = copy.deepcopy(refs['bbox'])
        gt_bbox[2] = gt_bbox[0] + gt_bbox[2]
        gt_bbox[3] = gt_bbox[1] + gt_bbox[3]
        gt_bbox_iter = np.array([gt_bbox])  # [1, 4]

        scores_iter, scores_mask_iter, transformed_bbox_iter, bbox_mask_iter = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
        if self.RUN_MODE in ['train']:
            scores_iter, scores_mask_iter, transformed_bbox_iter, bbox_mask_iter = self.proc_bbox_label(refs, frcn_feat)


        return torch.from_numpy(frcn_feat_iter),\
               torch.from_numpy(bbox_feat_iter),\
               torch.from_numpy(query_ix_iter),\
               torch.from_numpy(scores_iter).float(),\
               torch.from_numpy(scores_mask_iter).float(),\
               torch.from_numpy(transformed_bbox_iter).float(),\
               torch.from_numpy(bbox_mask_iter).float().unsqueeze(-1),\
               torch.from_numpy(gt_bbox_iter).float(),\
               torch.from_numpy(bbox_iter).float(),\
               torch.from_numpy(img_shape_iter).float(),\


    def proc_query(self, refs, max_token, tokenizer):
        if tokenizer is not None:
            query_words = refs['tokens']
            encoded_dict = tokenizer.encode_plus(
                query_words,
                add_special_tokens=True,
                max_length=max_token,  # Pad and truncate all questions
                return_tensors='pt',
                pad_to_max_length=True
            )
            query_ix = encoded_dict['input_ids']
            query_ix = torch.squeeze(query_ix).numpy()
        else:
            query_ix = np.zeros(max_token+1, np.int64)
            query_words = refs['tokens']
            for ix, word in enumerate(query_words):
                if word in self.token_to_ix:
                    query_ix[ix] = self.token_to_ix[word]
                else:
                    query_ix[ix] = self.token_to_ix['NOTFOUND']
                if ix+1 == max_token:
                    break

        return query_ix


    def proc_img_feat(self, img_feat, img_feat_pad_size):
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]

        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return img_feat


    def proc_bbox_feat(self, bbox, img_shape):
        bbox_feat = np.zeros((bbox.shape[0], 5), dtype=np.float32)
        bbox_feat[:, 0] = bbox[:, 0] / float(img_shape[1])
        bbox_feat[:, 1] = bbox[:, 1] / float(img_shape[0])
        bbox_feat[:, 2] = bbox[:, 2] / float(img_shape[1])
        bbox_feat[:, 3] = bbox[:, 3] / float(img_shape[0])
        bbox_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / float(img_shape[0] * img_shape[1])

        return bbox_feat


    def get_sigmoid_score(self, overlap, overlap_scores_threshold):
        if overlap < overlap_scores_threshold:
            return .0
        elif overlap < .6:
            return .8
        elif overlap < .7:
            return .9
        else:
            return 1.
        # return overlap


    def proc_bbox_label(self, refs, imgfeat_load):
        refs_bbox = copy.deepcopy(refs['bbox'])
        refs_bbox[2] = refs_bbox[0] + refs_bbox[2]
        refs_bbox[3] = refs_bbox[1] + refs_bbox[3]
        refs_bbox = np.array([refs_bbox])       # [1, 4]
        imgfeat_bbox = imgfeat_load['bbox']     # [N, 4]
        # print(refs_bbox, refs_info['width'], refs_info['height'])
        # print(imgfeat_bbox[0], imgfeat_load['image_w'], imgfeat_load['image_h'])
        # print(refs_bbox).shape)
        # print(imgfeat_bbox.shape)
        # overlaps = bbox_overlaps(imgfeat_bbox, refs_bbox)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(imgfeat_bbox, dtype=np.float),
            np.ascontiguousarray(refs_bbox, dtype=np.float))[:, 0]
        # print(overlaps.max())

        scores = np.zeros(100, dtype=np.float32)
        scores_mask = np.zeros(1, dtype=np.float32)
        bbox_mask = np.zeros(100, dtype=np.float32)
        if overlaps.max() >= self.__C.OVERLAP_THRESHOLD:
            scores_mask[0] = 1
            if self.__C.SCORES_LOSS == 'kld':
                scores_ix = np.where(overlaps >= self.__C.OVERLAP_THRESHOLD)[0]
                for ix in scores_ix:
                    scores[ix] = overlaps[ix]
                scores = scores / (scores.sum() + 1e-8)
            elif self.__C.SCORES_LOSS == 'bce':
                scores_ix = np.where(overlaps >= self.__C.OVERLAP_THRESHOLD)[0]
                for ix in scores_ix:
                    scores[ix] = self.get_sigmoid_score(overlaps[ix], self.__C.OVERLAP_THRESHOLD)
            else:
                print('wrong in loss !')
            bbox_mask[np.where(overlaps >= self.__C.OVERLAP_THRESHOLD)[0]] = 1

        transformed_bbox = bbox_transform(imgfeat_bbox, refs_bbox)    # np.tile(refs_bbox, (imgfeat_bbox.shape[0], 1))
        # print(transformed_bbox)
        if self.__C.BBOX_NORM:
            transformed_bbox = ((transformed_bbox - np.array(self.__C.BBOX_NORM_MEANS)) / np.array(self.__C.BBOX_NORM_STDS))
        # print(transformed_bbox.shape)
        transformed_bbox = self.proc_img_feat(transformed_bbox, img_feat_pad_size=100)
        # print(transformed_bbox.shape)

        return scores, scores_mask, transformed_bbox, bbox_mask


    def __len__(self):
        return self.data_size
