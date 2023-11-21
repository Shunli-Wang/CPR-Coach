import torch
from torch.utils.data import DataLoader
from core.datasets.dataset_ImagineNet_FC import FC_DoubleTestDataset
from core.models.ImagineNet_FC import ImagineNet_FC
from core.accuracy import eval_mAP_mmitmAP

# #### Double Dataset & Dataloader
testSet = FC_DoubleTestDataset('Multi_STGCN')
test_loader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=2)

# #### Model
model = ImagineNet_FC(in_channels=1024, num_classes=14)
model.cuda()

# #### Pre-trained weights
pthFile = 'ckpt/ST-GCN_ImagineNet_FC/ST-GCN_pre-trained.pth'
weights = torch.load(pthFile)
model.load_state_dict(weights)

# #### Testing
model.eval()
scoreList, labelList = [], []
for i, (feat, label) in enumerate(test_loader):
    score = model(feat.to('cuda'))
    scoreList.append(score.detach().cpu().reshape(-1))
    labelList.append(label.detach().cpu().reshape(-1))

# #### Eval
eval_mAP_mmitmAP(scoreList, labelList)
