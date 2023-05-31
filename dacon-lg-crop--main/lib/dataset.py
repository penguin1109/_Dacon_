import torch
import torch.nn as nn

def make_rcnn_labels(json_path):
    label = {}
    info = read_json(json_path)['annotations']

    x1 = info['bbox'][0]['x']
    y1 = info['bbox'][0]['y']
    x2 = x1 + info['bbox'][0]['w']
    y2 = y1 + info['bbox'][0]['h']
    bbox = torch.FloatTensor([x1,x2,y1,y2])
    crop = info['crop']
    target = f"{crop}"
    target = torch.tensor(train_crop_encoder[target])
    label['boxes'] = bbox
    label['labels'] = target

    return label

# custom dataset 생성
# 데이터를 dictionary의 형태로 제공하는 것도 가능하다

class LGCropDataset(Dataset):
  def __init__(self,ids, labels = None, mode = 'train'):
    self.mode = mode
    self.base = '/content/drive/MyDrive/dacon/LGfarm/data/train'
    self.train_img_base = '/content/drive/MyDrive/dacon/LGfarm/data/train/JPG'
    self.test_img_base = '/content/drive/MyDrive/dacon/LGfarm/data/test/JPG'
    self.ids = ids #10000, 100001, 10002.. 이런 형태의 데이터별 id를 저장함
    self.csv_feature_dict = csv_feature_dict
    self.csv_feature_check = [0] * len(self.ids)
    self.csv_features = [None] * len(self.ids)
    self.max_len = cfg['max_len']
    self.label_encoder = train_label_encoder
    self.crop_encoder = train_crop_encoder
    self.disease_encoder = train_disease_encoder
    self.data_df = data_df
  def __len__(self):
    return len(self.ids)
  
  def __getitem__(self, idx):
    id = self.ids[idx]
    json_path = f'{self.base}/JSON/{id}.json'
    img_path = f'{self.base}/JPG/{id}.jpg'

    if self.csv_feature_check[idx] == 0:
      csv_path = f'{self.base}/CSV/{id}.csv'
      df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
      df = df.replace('-', 0)

      # MinMax Scaling
      for col in self.csv_feature_dict.keys():
        df[col] = df[col].astype(float) - self.csv_feature_dict[col][0]
        df[col] = df[col] / (self.csv_feature_dict[col][1] - self.csv_feature_dict[col][0])
      # zero padding
      pad = np.zeros((self.max_len, len(df.columns)))
      length = min(self.max_len, len(df))
      pad[-length:] = df.to_numpy()[-length:]
      # Transpose to Sequential Data
      csv_feature = pad.T
      self.csv_features[idx] = csv_feature
      self.csv_feature_check[idx] = 1
    else:
      csv_feature = self.csv_features[idx]
    
    img_name = id + '.jpg'
    if self.mode == 'train':
      img = cv2.imread(os.path.join(self.train_img_base, img_name))
    else:
      img = cv2.imread(os.path.join(self.test_img_base, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # RGB형태로 읽었으니 채널수는 당연히 3 -> 255로 나누어 이미지 픽셀의 범위를 0-1로 바꾼다.
    img = cv2.resize(img, dsize = cfg['resize'], interpolation = cv2.INTER_AREA) 
    
    img = img.astype(np.float32)/255 # changed to the range of 0-1 # 정규화를 ImageNet weight로 바꾸기
    img = np.transpose(img, (2,0,1)) # (w,h,c) -> (c,w,h) #ToTensorV2와 동일한 역할
  
    if self.mode == 'train':
      with open(json_path, 'r') as f:
        json_file = json.load(f)
      
      annotations = json_file['annotations']
      crop, disease, risk = annotations['crop'], annotations['disease'], annotations['risk']
      label = f"{crop}_{disease}_{risk}"
      crop = f"{crop}"
      disease = f"{disease}_{risk}"

      return {
          'img' : torch.tensor(img, dtype = torch.float32),
          'csv_feature' : torch.tensor(csv_feature, dtype = torch.float32),
          'label' : torch.tensor(self.label_encoder[label], dtype = torch.long),
          'crop' : torch.tensor(self.crop_encoder[crop], dtype = torch.long),
          'disease' : torch.tensor(self.disease_encoder[disease], dtype = torch.long),
          'rcnn_target' : make_rcnn_labels(json_path)
      }
    else:
      return {
          'img' : torch.tensor(img, dtype = torch.float32),
          'csv_feature' : torch.tensor(csv_feature, dtype = torch.float32)
      }