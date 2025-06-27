class Path:
    def __init__(self):
        self.DATASET_ROOT_PATH = '/home/Datasets/vgd/data/'
        self.IMGFEAT_ROOT_PATH = '/home/Datasets/vgd/data/'

        # BERT预训练模型路径
        #self.PRETRAINED_PATH = '/home/xuyangshuyi/Openvqa_xy/pretrained_models/roberta-base-eng/'
        self.PRETRAINED_PATH = '/home/xuyangshuyi/Openvqa_xy/pretrained_models/bert-base-uncased/'
        # self.PRETRAINED_PATH = '/home/xuyangshuyi/Openvqa_xy/pretrained_models/albert-base-v2-uncased/'

        self.CKPT_PATH = './logs/ckpts/'

        self.IMGFEAT_PATH = {
            'vg_woref':{
                'train': self.IMGFEAT_ROOT_PATH + 'bua-r101-fix100/',
            },
            # 'coco_mrcn':{
            #     'refcoco': self.IMGFEAT_ROOT_PATH + 'vgd_coco/fix100/refcoco_unc/',
            #     'refcoco+': self.IMGFEAT_ROOT_PATH + 'vgd_coco/fix100/refcoco+_unc/',
            #     'refcocog': self.IMGFEAT_ROOT_PATH + 'vgd_coco/fix100/refcocog_umd/',
            # },
        }

        self.REF_PATH = {
            'refcoco': {
                'train': self.DATASET_ROOT_PATH + 'refcoco/train.json',
                'val': self.DATASET_ROOT_PATH + 'refcoco/val.json',
                'testA': self.DATASET_ROOT_PATH + 'refcoco/testA.json',
                'testB': self.DATASET_ROOT_PATH + 'refcoco/testB.json',
            },
            'refcoco+': {
                'train': self.DATASET_ROOT_PATH + 'refcoco+/train.json',
                'val': self.DATASET_ROOT_PATH + 'refcoco+/val.json',
                'testA': self.DATASET_ROOT_PATH + 'refcoco+/testA.json',
                'testB': self.DATASET_ROOT_PATH + 'refcoco+/testB.json',
            },
            'refcocog': {
                'train': self.DATASET_ROOT_PATH + 'refcocog/train.json',
                'val': self.DATASET_ROOT_PATH + 'refcocog/val.json',
                'test': self.DATASET_ROOT_PATH + 'refcocog/test.json',
            },
        }

        self.EVAL_PATH = {
            'result_test': self.CKPT_PATH + 'result_test/',
            'tmp': self.CKPT_PATH + 'tmp/',
            'arch': 'arch/',
        }
