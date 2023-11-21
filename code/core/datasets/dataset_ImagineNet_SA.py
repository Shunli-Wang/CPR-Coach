import torch
import numpy as np
import pickle
import random
from random import sample
from torch.utils.data import Dataset

class SA_ComposeDataset(Dataset):
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
        self.labelList, self.featList = [], []
        
        # 2
        from itertools import combinations
        combin2 = list(combinations([i for i in range(1, 14)], 2)) # [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), ...
        combin2 = sample(combin2, int(len(combin2)/2))
        # Store all fake data
        for combinElem in combin2:
            crossIdxList1 = self.idxCls[combinElem[0]]
            crossIdxList2 = self.idxCls[combinElem[1]]
            for i in range(len(self.idxCls[0])):
                selectedIdx1 = random.randint(0, len(crossIdxList1)-1)
                selectedIdx2 = random.randint(0, len(crossIdxList2)-1)
                idx1, idx2 = crossIdxList1[selectedIdx1], crossIdxList2[selectedIdx2]
                feat1, feat2 = self.allData[idx1], self.allData[idx2]
                randWeight = np.random.rand() # get random weight
                self.labelList.append(list(combinElem))

                randWeightEnable = 1 # 0 for disable; 1 for enable
                
                if randWeightEnable == 0:
                    self.featList.append([feat1, feat2]) # Add
                elif randWeightEnable == 1:
                    self.featList.append([feat1 * randWeight, feat2 * (1 - randWeight)]) # Split

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
            feat = self.featList[index][0] + self.featList[index][1]
            onehot = torch.zeros(14)
            onehot[self.labelList[index]] = 1.
            label = onehot
        except:
            print('feature id %d not found' % index)
            return None

        return feat, label

class SA_DoubleTestDataset(Dataset):
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
