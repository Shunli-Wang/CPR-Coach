import torch
import numpy as np
import pickle
import random
from itertools import combinations
from torch.utils.data import Dataset 

class CA_ComposeDataset(Dataset):
    ''' Compose features from 13 classes. '''
    def __init__(self, featDataName):
        # Load feature
        fullFileName = './pkl/' + featDataName + '.pkl'
        with open(fullFileName, 'rb') as f:
            self.data = pickle.load(f)
        self.allLabel = np.array([i[0][0] for i in self.data]) # (896, )
        self.allData = np.array([i[1] for i in self.data]) # (8, 2048)

        self.initDataset()

    def initDataset(self):

        # Find all cls idx
        self.idxCls = [np.where(self.allLabel == i)[0] for i in range(14)]

        # Combine
        combin = list(combinations([i for i in range(1, 14)], 2)) # [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), ...

        # Store all fake data
        self.labelList, self.featList = [], []

        for combinElem in combin:
            # Random select a clothing, cross under the same suit
            dataCeossMode = 1 # 0 for intra-cross; 1 for inter-cross

            if dataCeossMode == 0:
                # #### Select:
                suit = random.randint(0, 1)
                idxLen = int(self.idxCls[0].shape[0] / 2) # self.idxCls[0].shape[0]
                if suit == 0:
                    crossIdxList1 = self.idxCls[combinElem[0]][0:idxLen]
                    crossIdxList2 = self.idxCls[combinElem[1]][0:idxLen]
                else:
                    crossIdxList1 = self.idxCls[combinElem[0]][idxLen:]
                    crossIdxList2 = self.idxCls[combinElem[1]][idxLen:]
            elif dataCeossMode == 1:
                # #### Cross suit
                idxLen = self.idxCls[0].shape[0]
                crossIdxList1 = self.idxCls[combinElem[0]]
                crossIdxList2 = self.idxCls[combinElem[1]]

            for i in range(2 * int(idxLen)):
                selectedIdx1 = random.randint(0, idxLen-1)
                selectedIdx2 = random.randint(0, idxLen-1)
                idx1, idx2 = crossIdxList1[selectedIdx1], crossIdxList2[selectedIdx2]
                feat1, feat2 = self.allData[idx1], self.allData[idx2]
                
                randWeight = np.random.rand() # get random weight

                self.labelList.append(list(combinElem))

                randWeightEnable = 0 # 0 for disable; 1 for enable
                if randWeightEnable == 0:
                    self.featList.append([feat1, feat2])
                elif randWeightEnable == 1:
                    self.featList.append([feat1 * randWeight, feat2 * (1 - randWeight)])

    def __len__(self):
        return len(self.labelList)
    
    def __getitem__(self, index):
        item = self.getitem1(index)
        while item is None:
            index = random.randint(0, len(self.data) - 1)
            item = self.getitem1(index)

        return item
    
    def getitem1(self, index):
        try:
            feat = self.featList[index] # Split
            # feat = self.featList[index][0] + self.featList[index][1] # [(2048, ), (2048, )] # Add
            onehot = torch.zeros(14)
            onehot[self.labelList[index]] = 1.
            label = onehot
        except:
            print('feature id %d not found' % index)
            return None

        return feat, label

class CA_DoubleTestDataset(Dataset):
    ''' This is used for double-class testing. '''
    def __init__(self, featDataName):
        fullFileName = './pkl/' + featDataName + '.pkl'
        with open(fullFileName, 'rb') as f:
            self.data = pickle.load(f)
        self.allLabel = np.array([i[0].reshape(-1) for i in self.data])
        self.allData = np.array([i[1] for i in self.data])

        print('Loading data: ', len(self.data))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.getitem1(index)
        while item is None:
            index = random.randint(0, len(self.data) - 1)
            item = self.getitem1(index)

        return item
    
    def getitem1(self, index):
        try:
            feat = self.allData[index]
            label = self.allLabel[index]
        except:
            print('feature id %d not found' % index)
            return None

        return feat, label

class RGB_Pose_ComposeDataset(Dataset):
    ''' Compose features from 13 classes. '''
    def __init__(self, TSN_Name, STGCN_Name):
        # Load ST-GCN feature
        STGCN_Name = './pkl/' + STGCN_Name + '.pkl'
        with open(STGCN_Name, 'rb') as f:
            self.data = pickle.load(f)
        self.allLabelPose = np.array([i[0][0] for i in self.data]) # (896, )
        self.allDataPose = np.array([i[1] for i in self.data]) # (896, 512)

        # Load TSN feature
        TSN_Name = './pkl/' + TSN_Name + '.pkl'
        with open(TSN_Name, 'rb') as f:
            self.data = pickle.load(f)
        self.allLabelRGB = np.array([i[0][0] for i in self.data]) # (896, )
        self.allDataRGB = np.array([i[1] for i in self.data]) # (896, 8, 2048)

        self.initDataset()

    def initDataset(self):

        # Find all cls idx
        self.idxCls = [np.where(self.allLabelPose == i)[0] for i in range(14)]

        # Combine
        combin = list(combinations([i for i in range(1, 14)], 2)) # [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), ...

        # Store all fake data
        self.labelList = []
        self.featList = []

        for combinElem in combin:
            # Random select a clothing, cross under the same suit
            dataCeossMode = 1 # 0 for intra-cross; 1 for inter-cross

            if dataCeossMode == 0:
                # #### Select:
                suit = random.randint(0, 1)
                idxLen = int(self.idxCls[0].shape[0] / 2) # self.idxCls[0].shape[0]
                if suit == 0:
                    crossIdxList1 = self.idxCls[combinElem[0]][0:idxLen]
                    crossIdxList2 = self.idxCls[combinElem[1]][0:idxLen]
                else:
                    crossIdxList1 = self.idxCls[combinElem[0]][idxLen:]
                    crossIdxList2 = self.idxCls[combinElem[1]][idxLen:]
            elif dataCeossMode == 1:
                # #### Cross suit
                idxLen = self.idxCls[0].shape[0]
                crossIdxList1 = self.idxCls[combinElem[0]]
                crossIdxList2 = self.idxCls[combinElem[1]]

            for i in range(2 * int(idxLen)):
                selectedIdx1 = random.randint(0, idxLen-1)
                selectedIdx2 = random.randint(0, idxLen-1)
                idx1, idx2 = crossIdxList1[selectedIdx1], crossIdxList2[selectedIdx2]
                feat1, feat2 = self.allDataRGB[idx1], self.allDataPose[idx2]
                
                randWeight = np.random.rand() # get random weight

                self.labelList.append(list(combinElem))

                randWeightEnable = 1 # 0 for disable; 1 for enable
                if randWeightEnable == 0:
                    self.featList.append([feat1, feat2])
                elif randWeightEnable == 1:
                    self.featList.append([feat1 * randWeight, feat2 * (1 - randWeight)])

    def __len__(self):
        return len(self.labelList)
    
    def __getitem__(self, index):
        item = self.getitem1(index)
        while item is None:
            index = random.randint(0, len(self.data) - 1)
            item = self.getitem1(index)

        return item
    
    def getitem1(self, index):
        try:
            feat = [self.featList[index][0], torch.tensor(self.featList[index][1]).repeat((1, 4)).squeeze().repeat((8, 1))] # Split
            onehot = torch.zeros(14)
            onehot[self.labelList[index]] = 1.
            label = onehot
        except:
            print('feature id %d not found' % index)
            return None

        return feat, label

class RGB_Pose_DoubleTestDataset(Dataset):
    ''' This is used for double-class testing. '''
    def __init__(self, TSN_Name, STGCN_Name):

        # Load ST-GCN feature
        STGCN_Name = './pkl/' + STGCN_Name + '.pkl'
        with open(STGCN_Name, 'rb') as f:
            self.data = pickle.load(f)
        self.allLabelPose = np.array([i[0].reshape(-1) for i in self.data])  # (2358, 14)
        self.allDataPose = np.array([i[1] for i in self.data]) # (2358, 512)

        # Load TSN feature
        TSN_Name = './pkl/' + TSN_Name + '.pkl'
        with open(TSN_Name, 'rb') as f:
            self.data = pickle.load(f)
        self.allLabelRGB = np.array([i[0].reshape(-1) for i in self.data])  # (2360, 14)
        self.allDataRGB = np.array([i[1] for i in self.data]) # (2360, 2048)

        # Filter out 2 null samples
        for i in range(len(self.allLabelPose)):
            if 1 in self.allLabelPose[i] + self.allLabelRGB[i]:
                self.allLabelRGB = np.delete(self.allLabelRGB, i, axis = 0)
        
    def __len__(self):
        return len(self.allLabelRGB)
    
    def __getitem__(self, index):
        item = self.getitem1(index)
        while item is None:
            index = random.randint(0, len(self.data) - 1)
            item = self.getitem1(index)

        return item
    
    def getitem1(self, index):
        try:
            feat = [self.allDataRGB[index], torch.tensor(self.allDataPose[index]).repeat((1, 4)).squeeze().repeat((8, 1))]
            label = self.allLabelRGB[index]
        except:
            print('feature id %d not found' % index)
            return None

        return feat, label
