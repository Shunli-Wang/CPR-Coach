import torch
from torch import optim
from torch.utils.data import DataLoader
from core.datasets.dataset_ImagineNet_FC import FC_ComposeDataset, FC_DoubleTestDataset
from core.models.ImagineNet_FC import ImagineNet_FC
from core.accuracy import eval_mAP_mmitmAP

# Model Name
name = 'STGCN_ImagineNet_FC'

# #### Single Dataset & Dataloader
trainSet = FC_ComposeDataset('Single_STGCN')
train_loader = DataLoader(trainSet, batch_size=8, shuffle=True, num_workers=2)

# #### Double Dataset & Dataloader
testSet = FC_DoubleTestDataset('Multi_STGCN')
test_loader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=2)

# #### Model
model = ImagineNet_FC(in_channels=1024, num_classes=14)
model.cuda()

# #### Optimizer
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

# ####
# #### Training 
# ####
for epoch in range(60):
    model.train()
    for i, (feat, label) in enumerate(train_loader):
        model.zero_grad()
        score, loss = model(feat.to('cuda'), label.to('cuda'))  # feat [8, 2048]

        loss.backward()
        optimizer.step()

    trainSet.initDataset()
    scheduler.step()
    
    if (epoch + 1) % 5 == 0:
        print('epoch: %02d, lr: %f, loss: %f ' % (epoch + 1, optimizer.state_dict()['param_groups'][0]['lr'], loss), end='')
        # ####
        # #### Testing
        # ####
        scoreList, labelList = [], []
        model.eval()
        for i, (feat, label) in enumerate(test_loader):
            score = model(feat.to('cuda'))
            scoreList.append(score.detach().cpu().reshape(-1))
            labelList.append(label.detach().cpu().reshape(-1))
        eval_mAP_mmitmAP(scoreList, labelList)
        torch.save(model.state_dict(), 'ckpt/ST-GCN_ImagineNet_FC/%02d_ST-GCN.pth'%(epoch + 1))
