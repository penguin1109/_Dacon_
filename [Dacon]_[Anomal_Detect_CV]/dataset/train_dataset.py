import torchvision
import torchvision.transforms as transform

TEXTURE = ['wood', 'tile', 'grid', 'carpet', 'leather']

IMG_SIZE = 600

def texture_transform_anormal():
  aug = transform.Compose([
      transforms.ToPILImage(),
      transforms.Resize(IMG_SIZE),
      transforms.ToTensor(),
      transforms.RandomChoice([
          transforms.RandomVerticalFlip(),
          transforms.RandomHorizontalFlip(),
          transforms.RandomRotation(90),
      ]),
                    transforms.ColorJitter(brightness =(0.5,1.5)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  return aug

def texture_transform_normal():
  aug = transform.Compose([
      transforms.ToPILImage(),
      #transforms.RandomSizedCrop(456),
      transforms.Resize(IMG_SIZE),
      transforms.ToTensor(),
      transforms.RandomChoice([
          transforms.RandomVerticalFlip(),
          transforms.RandomHorizontalFlip(),
          transforms.RandomRotation(90),
      ]),
                      transforms.ColorJitter(brightness =(0.5,2.0)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  return aug

def object_transform():
  aug = transform.Compose([
                           
                           transforms.ToPILImage(),
                           transforms.Resize(IMG_SIZE),
                           transforms.ToTensor(),
                           transforms.RandomChoice(
                               [
                                transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(90),
                               ]
                           ),
                           transforms.RandomAutocontrast(p = 0.4),
                           transforms.ColorJitter(brightness =(0.5,2.0)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

  ])
  return aug

def val_transform():
  aug =  transform.Compose([
                            
                            transforms.ToPILImage(),
                            transforms.Resize(IMG_SIZE),
                            transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  return aug

class TrainValDataset(Dataset):
  def __init__(self, img_pths, classes, states, labels, mode = 'train'):
    self.img_pths = img_pths
    self.classes = classes
    self.states = states
    self.labels = labels
    self.mode = mode
  
  def __len__(self):
    return len(self.img_pths)
  
  def __getitem__(self, idx):
    img = self.img_pths[idx]
    img = cv2.imread(img)
    # img = PIL.Image.fromarray(img)
  
    state = state_encoder[self.states[idx]]
    cls = class_encoder[self.classes[idx]]
 
    label = self.labels[idx]
    mask = anomal_mask[self.classes[idx]]

    if self.mode == 'train':
      if self.classes[idx] in TEXTURE:
        if state == 'good':
          aug = texture_transform_normal()
        else:
          aug = texture_transform_anormal()
      else:
        aug = object_transform()
    else:aug = val_transform()
    img = aug(img)

    return img, state, cls, label, torch.tensor(mask)
