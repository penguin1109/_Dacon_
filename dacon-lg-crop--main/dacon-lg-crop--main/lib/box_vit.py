import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms.functional as TF

import torch
import torch.nn as nn

from vit_layers import *

# Faster RCNN model used to predict the target crop's bounding box
# and the type of crop the image indicates
# 타겟 작물을 crop 하는 작업을 학습 시킨 faster rcnn의 사전학습된 weigth를 load해서 넣어줌
def build_frcnn(path):
    backbone = torchvision.models.vgg16(pretrained=True).features[:-1]
    backbone_out = 512
    backbone.out_channels = backbone_out
    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(sizes=((128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))
    resolution = 7
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=resolution, sampling_ratio=2)

    model = FasterRCNN(backbone, num_classes = 1+6, rpn_anchor_generator = anchor_generator,box_detections_per_img=1, box_roi_pool = roi_pooler) #num classes should be including the background
    model.load_state_dict(torch.load(path))
    
    return model

class LG_Crop(nn.Module):
    def __init__(self, channel_in = 3,
                 patch_size = 16,
                 emb_size = 1280,
                 img_size = (384, 512),
                 depth = 16,
                 max_len = 24 * 6,
                 num_features = cfg['num_feats']
                 **kwargs):
        super().__init__()
        self.rcnn = build_frcnn()
        self.cnn = models.resnet50(pretrained = True)
        self.rnn = RNNDecoder(depth, emb_size, num_features)
        self.emb = PatchEmbed(channel_in, patch_size, emb_size, img_size)
        self.enc = TransformerEncoder(depth, emb_size, **kwargs)
        self.disease_classifier = ClassificationHead(emb_size*2 + 1000, 17) # 17은 총 질병 라벨 개수
    
    def add_preds(self, crop, disease):
        crop_label = train_crop_decoder[crop]
        disease_label = train_disease_decoder[disease]
        final_label = f"{crop_label}_{disease_label}"
        final_predict = train_label_decoder[final_label]
        
        return final_predict
    
    def forward(self, img):
        rcnn_res = self.rcnn(img)
        boxes, labels, scores = rcnn_res['boxes'], rcnn_res['labels']. rcnn_res['scores']
        crop = labels
        
        top, left, height, width = boxes[:,1], boxes[:,0], boxes[:, 3] - boxes[:,1], boxes[:,2] - boxes[:,0]
        
        # crop the target image based on rcnn's predictions
        new_img = TF.crop(img,top, left, height, width)
        new_img = TF.resize(new_img, [224, 224])
        
        # predict the {disease}_{risk}
        cnn = self.cnn(new_img)
        enc = self.enc(self.emb(new_img))
        seq = self.rnn(seq)
        disease = self.disease_classifier(enc, seq, cnn)
        
        final = self.add_preds(crop, disease)
        
        return crop, disease, final
         
        
