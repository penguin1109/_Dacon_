## utils.py
"""
Includes the function for..
1. Generating a Bounding Box Using the OpenCV Contouring function
2. Custom Optimizer
3. Custom Learning Rate Scheduler
4. Generating the Prediction Based on the Class Classification
5. Generating the Mask for State Classification
"""
import torch
import numpy as np
## meta_data.py

anomal_mask = {}
for k, values in state_class_pair.items():
  anomal_mask[k] = [True] * len(states)
  for v in values:
    anomal_mask[k][states.index(v)] = False

def get_mask(class_logits):
    pred_cls = class_logits.softmax(dim = -1).detach().cpu().numpy().argmax(1).tolist()
    masks = []
    for idx, c in enumerate(pred_cls):
        mask = anomal_mask[class_decoder[c]]
        masks.append(mask)
    masks = np.array(masks)
    masks = torch.from_numpy(masks)
    return masks

def generate_prediction(class_logits, state_logits):
    pred_cls = class_logits.softmax(dim = -1).detach().cpu().numpy().argmax(1).tolist() ## 예측한 class의 종류를 확인할 수 있음
    pred_states = state_logits.softmax(dim = -1).detach().cpu().numpy().tolist()

    pred_label = []
    for idx, c in enumerate(pred_cls):
        cls = class_decoder[c]
        mask = anomal_mask[cls]
        state = pred_states[idx]
        for idx, t in enumerate(state):
          if mask[idx] == True:
            state[idx] = -10000
        pred_state = state_decoder[np.argmax(state)]
        pred_label.append(label_unique[cls + '-' + pred_state])

    return pred_label

def load_models(MODEL):

    CKPT_DIR = '/content/drive/MyDrive/dacon/Anomaly_detection_vision/ckpt'

    base_model = BaseModel(drop_p=BASE_DROP_RATE)
    base_model = torch.nn.DataParallel(base_model)
    if (os.path.isfile(os.path.join(CKPT_DIR, f'{MODEL}_base_1.pt'))):
        base_model.load_state_dict(torch.load(os.path.join(CKPT_DIR,  f'{MODEL}_base_1.pt')))
        print("BASE MODEL  LOADED")
    base_model = base_model.to(device)

    class_model = ClassHeadClassifier(drop_p=CLASS_DROP_RATE)
    class_model = torch.nn.DataParallel(class_model)
    if (os.path.isfile(os.path.join(CKPT_DIR,  f'{MODEL}_cls_1.pt'))):
        class_model.load_state_dict(torch.load(os.path.join(CKPT_DIR,  f'{MODEL}_cls_1.pt')))
    print("CLASS MODEL LOADED")
    class_model = class_model.to(device)

    anomal_model = AnomalHeadClassifier(drop_p=ANOMAL_DROP_RATE)
    anomal_model = torch.nn.DataParallel(anomal_model)
    if (os.path.isfile(os.path.join(CKPT_DIR,  f'{MODEL}_anomal_1.pt'))):
        anomal_model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f'{MODEL}_anomal_1.pt')))
    print("ANOMAL MODEL LOADED")
    anomal_model = anomal_model.to(device)
    
    return base_model, class_model, anomal_model

## Focal Loss.py

import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=5, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    @torch.cuda.amp.autocast()
    def forward(self, inputs, targets, mixup=None):
        loss = self.loss_fn(inputs, targets)
        pt = torch.exp(-loss)
        F_loss = self.alpha * (1-pt)**self.gamma * loss
        
        if mixup is not None:
            F_loss = F_loss * mixup
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        elif self.reduction == 'none':
            return F_loss