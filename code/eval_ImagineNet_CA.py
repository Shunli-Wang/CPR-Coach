import torch
from torch.utils.data import DataLoader
from core.datasets.dataset_ImagineNet_CA import CA_DoubleTestDataset
from core.models.ImagineNet_CA import ImagineNet_CA
from core.accuracy import eval_mAP_mmitmAP

# #### Double-error Dataset & Dataloader
testSet = CA_DoubleTestDataset('TSN_Double_Feature')
test_loader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=2)

# #### Model
model = ImagineNet_CA(in_channels=2048, num_classes=14)
model.cuda()

# #### Pre-trained weights
pthFile = '/PATH/TO/ImagineNet_CA.pth'
weights = torch.load(pthFile)
model.load_state_dict(weights)

# #### Testing
scoreList, labelList = [], []
model.eval()
for i, (feat, label) in enumerate(test_loader):
    score = model(feat.to('cuda'), feat.to('cuda'))
    scoreList.append(score.detach().cpu().reshape(-1))
    labelList.append(label.detach().cpu().reshape(-1))

# #### Eval
eval_mAP_mmitmAP(scoreList, labelList)
