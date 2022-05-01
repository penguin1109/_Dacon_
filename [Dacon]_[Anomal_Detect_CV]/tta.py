"""
TTA (Test Time Augmentation)

"""

import ttach as tta
from torch.utils.data import DataLoader
from glob import glob
from collections import Counter

from test_dataset import TestDataset
from models import *
from utils import *

test_png = sorted(glob('/content/drive/MyDrive/dacon/Anomaly_detection_vision/data/test/*.png'))
test_dataset = TestDataset(np.array(test_png))
test_loader = DataLoader(test_dataset, shuffle = False, batch_size = cfg.BATCH_SIZE)

transforms = tta.Compose([
    tta.HorizontalFlip(),
    tta.VerticalFlip(),
    tta.Rotate90(angles = [0,90]),
    tta.Multiply(factors=[0.7, 1]),
])

base_model, cls_model, anomal_model = get_models(cfg.MODEL)
base_model.eval();cls_model.eval();anomal_model.eval();

predictions = []
with torch.no_grad():
    for img in test_loader:
        preds = []
        img = img.to(cfg.DEVICE)
        for i, transformer in eumerate(transforms):
            augments = transformer.augment_image(img)
            
            img_feat = base_model(augments)
            cls_pred = cls_model(img_feat)
            mask = get_mask(cls_pred).to(cfg.DEVICE)
            anomal_pred = anomal_model(img_feat, mask)
            preds.append(generate_prediction(cls_pred, anomal_pred))
        predictions.append(Counter(preds).most_common(1))

label_decoder = {val:key for (key, val) in label_unique.items()}
results = [label_decoder[result] for result in predictions]
submission = pd.read_csv('/content/drive/MyDrive/dacon/Anomaly_detection_vision/data/sample_submission.csv')

submission["label"] = f_result
submission.to_csv(os.path.join('/content/drive/MyDrive/dacon/Anomaly_detection_vision/data',f"prediction_{cfg.MODEL}.csv"), index = False)
    