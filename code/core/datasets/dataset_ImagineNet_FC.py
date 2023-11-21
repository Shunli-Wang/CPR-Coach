import torch
import numpy as np
import pickle
import random
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader

class FC_ComposeDataset(Dataset):
    ''' Compose features from 13 classes. '''
    def __init__(self, featDataName):
        # Load feature
        fullFileName = './pkl/' + featDataName + '.pkl'
        with open(fullFileName, 'rb') as f:
            self.data = pickle.load(f)
        self.allLabel = np.array([i[0][0] for i in self.data]) # (896, )
        self.allData = np.array([i[1] for i in self.data])

        self.initDataset()

    def initDataset(self):

        # Find all cls idx
        self.idxCls = [np.where(self.allLabel == i)[0] for i in range(14)]

        # Combine
        from itertools import combinations
        combin = list(combinations([i for i in range(1, 14)], 2)) # [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), ...

        # Store all fake data
        self.labelList = []
        self.featList = []

        for combinElem in combin:
            # Random select a clothing, cross under the same suit
            # #### Select:
            # suit = random.randint(0, 1)
            # idxLen = int(self.idxCls[0].shape[0] / 2) # self.idxCls[0].shape[0]
            # if suit == 0:
            #     crossIdxList1 = self.idxCls[combinElem[0]][0:idxLen]
            #     crossIdxList2 = self.idxCls[combinElem[1]][0:idxLen]
            # else:
            #     crossIdxList1 = self.idxCls[combinElem[0]][idxLen:]
            #     crossIdxList2 = self.idxCls[combinElem[1]][idxLen:]
            
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
                # self.featList.append([feat1 * randWeight, feat2 * (1 - randWeight)]) # with rand
                self.featList.append([feat1, feat2])

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
            # feat = self.featList[index]
            feat = self.featList[index][0] + self.featList[index][1] # [(2048, ), (2048, )]
            
            onehot = torch.zeros(14)
            onehot[self.labelList[index]] = 1.
            label = onehot
        except:
            print('feature id %d not found' % index)
            return None

        return feat, label

class FC_DoubleTestDataset(Dataset):
    ''' This is used for double-class testing. '''
    def __init__(self, featDataName):
        fullFileName = './pkl/' + featDataName + '.pkl'
        with open(fullFileName, 'rb') as f:
            self.data = pickle.load(f)
        self.allLabel = np.array([i[0].reshape(-1) for i in self.data])
        self.allData = np.array([i[1] for i in self.data])

        print('Loading data: ', len(self.data))
        print('label shape: ', self.allLabel.shape)
        print('Feature shape: ', self.allData.shape)

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
