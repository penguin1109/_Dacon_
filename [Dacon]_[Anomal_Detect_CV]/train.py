 ## 우선은 validaation 없이 훈련만 시켜보도록 하자
from tqdm import tqdm
import PIL
def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score

torch.cuda.empty_cache()
MODE = 'both'
SCHEDULER = True
FINETUNE = False
SE = False
MODEL = 'effb7'


BASE_DROP_RATE = 0.0
CLASS_DROP_RATE = 0.1
ANOMAL_DROP_RATE = 0.1
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
#full_model = FullHeadClassifier(drop_p=0.3)
#full_model = torch.nn.DataParallel(full_model).to(device)
for param in base_model.parameters():
  param.requires_grad = True
for param in class_model.parameters():
  param.requires_grad = True


if MODEL != 'res':
  if (MODE == 'class_only'):
    optimizer = torch.optim.AdamW([
      {'params' : base_model.parameters()},
        {'params': class_model.parameters()}], lr= 3e-4, weight_decay=0.01)
  else:
    optimizer = torch.optim.AdamW([
        {'params' : base_model.parameters(), 'lr' : 1e-4, 'weight_decay' : 0.009},
        {'params': class_model.parameters(), 'lr' : 2e-4, 'weight_decay' : 0.009}, 
        {'params': anomal_model.parameters(), 'lr' : 3e-4, 'weight_decay' : 0.01}])
else:
  optimizer = torch.optim.SGD([
    {'params' : base_model.parameters()},
    {'params' : class_model.parameters()},
    {'params' : anomal_model.parameters(),}], lr = 1e-2,momentum = 0.09, 
  )
## 모델의 loss가 커진다면 이는 anomal state를 판단하는 것에 문제가 있을 확률이 높을 것으로 판단하였다.

class_fn = FocalLoss(gamma = 1)
# anomal_fn = FocalLoss(ignore_index = 0)
anomal_fn = FocalLoss(gamma = 2) ## ignore index는 아직 하지 않는 것이 나아 보인다. -> Focal Loss를 하면 잘 예측 못하는 class에 대해서 가중치를 높여준다.

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 15, eta_min = 1e-4)
scaler = torch.cuda.amp.GradScaler() 

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(os.path.join(DATA_ROOT, 'log'))

class_best, anomal_best = 0, 0
best = 0
for epoch in range(epochs):
    if (epoch >= 40):
      for name, p in base_model.named_parameters():
        if not (isinstance(p, nn.BatchNorm2d)):
          #p.bias.requires_grad = False
          #p.weight.requires_grad = False
          p.requires_grad = False
    if (epoch >= 10):
      schedler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, min_lr = 1e-4)
    start=time.time()
    train_loss = 0
    loss = 0.0
    train_pred, train_y = [], []
    base_model.train()

    class_model.train()
    anomal_model.train()
    
    t = tqdm(train_loader, desc = f"EPOCH : {epoch+1}")
    pred_cls, pred_final, pred_state = [], [], []
    torch.cuda.empty_cache()

    for (img, state, cls, label, mask) in t:
        if (epoch >= 30):
          scheduler.step()
        optimizer.zero_grad()
        img = img.to(device)

        state = torch.tensor(state, dtype = torch.long, device = device)
        label = torch.tensor(label, dtype = torch.long, device = device)
        cls = torch.tensor(cls, dtype = torch.long, device = device)
        mask = torch.tensor(mask,device = device)

        with torch.cuda.amp.autocast():
            img_feats = base_model(img)
            cls_logits = class_model(img_feats)
            state_logits = anomal_model(img_feats, mask)
            cls_loss, anomal_loss = class_fn(cls_logits, cls), anomal_fn(state_logits, state)
            if (MODE == 'class_only'):
              loss = cls_loss
            else:
              loss = (cls_loss + anomal_loss)
            
            pred_cls.extend(cls_logits.softmax(dim = -1).detach().cpu().numpy().argmax(1).tolist()) ## 어떤 물체 부분인지 classification 하는 부분
            pred_state.extend(state_logits.softmax(dim = -1).detach().cpu().numpy().argmax(1).tolist()) ## 어떤 anomal 한 state인지 classification 하는 부분
            
            if (MODE == 'class_only'):
              t.set_postfix({'cls loss' : cls_loss.item()})
            else:
              t.set_postfix({'loss' : loss.item(), 'cls loss' : cls_loss.item(), 'anomal loss' : anomal_loss.item()})
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()/len(train_loader)
        if (MODE == 'class_only'):
          pred_final = pred_cls
          train_y += cls.detach().cpu().numpy().tolist()
        else:
          pred_final += generate_prediction(cls_logits, state_logits)
          train_y += label.detach().cpu().numpy().tolist()
        
    
    train_f1 = score_function(train_y, pred_final)
    if (train_f1 > best):
      best = train_f1
      if FINETUNE == True:
        torch.save(base_model.state_dict(), os.path.join(CKPT_DIR, f'{MODEL}_base_FINETUNE_1.pt'))
        torch.save(class_model.state_dict(), os.path.join(CKPT_DIR,  f'{MODEL}_cls_FINETUNE_1.pt'))
        torch.save(anomal_model.state_dict(), os.path.join(CKPT_DIR,  f'{MODEL}_anomal_FINETUNE_1.pt'))
      else:
        torch.save(base_model.state_dict(), os.path.join(CKPT_DIR,  f'{MODEL}_base_1.pt'))
        torch.save(class_model.state_dict(), os.path.join(CKPT_DIR,  f'{MODEL}_cls_1.pt'))
        torch.save(anomal_model.state_dict(), os.path.join(CKPT_DIR, f'{MODEL}_anomal_1.pt'))


    TIME = time.time() - start
    print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s/{TIME*(epochs-epoch-1):.0f}s')
    print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')