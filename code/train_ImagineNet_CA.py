import torch
from torch import optim
from torch.utils.data import DataLoader
from core.datasets.dataset_ImagineNet_CA import CA_ComposeDataset, CA_DoubleTestDataset
from core.models.ImagineNet_CA import ImagineNet_CA
from core.accuracy import eval_mAP_mmitmAP

# Model Name
name = 'TSN_ImagineNet_CA'

# #### Single-error Dataset & Dataloader
trainSet = CA_ComposeDataset('TSN_118_Corp(1,0.875,0.75)')
train_loader = DataLoader(trainSet, batch_size=8, shuffle=True, num_workers=2)

# #### Double-error Dataset & Dataloader
testSet = CA_DoubleTestDataset('TSN_Double_Feature')
test_loader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=2)

# #### Model
model = ImagineNet_CA(in_channels=2048, num_classes=14)
model.cuda()

# #### Optimizer
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9, weight_decay=0.0001, nesterov=True,)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

# ####
# #### Training
# ####
for epoch in range(60):
    model.train()
    for i, (feat, label) in enumerate(train_loader):
        model.zero_grad()
        score, loss = model(feat[0].to('cuda'), feat[1].to('cuda'), label.to('cuda'))  # feat [8, 8, 2048]

        loss.backward()
        optimizer.step()

    trainSet.initDataset()
    scheduler.step()
    print('epoch: %02d, lr: %f, loss: %.4f, ' % (epoch, optimizer.state_dict()['param_groups'][0]['lr'], loss), end='')

    if (epoch + 1) % 5 == 0:
        # ####
        # #### Testing
        # ####
        scoreList, labelList = [], []
        model.eval()
        for i, (feat, label) in enumerate(test_loader):
            score = model(feat.to('cuda'), feat.to('cuda'))
            scoreList.append(score.detach().cpu().reshape(-1))
            labelList.append(label.detach().cpu().reshape(-1))

        # #### Eval
        eval_mAP_mmitmAP(scoreList, labelList)

        torch.save(model.state_dict(), 'ckpt/TSN_ImagineNet_CA/%02d_TSN.pth'%(epoch))
