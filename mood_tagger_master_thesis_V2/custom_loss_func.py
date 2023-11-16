from denseweight import DenseWeight
from torch import nn, optim
import torch
import numpy as np

class dense_weight_loss_single(nn.Module):
    # cite https://github.com/SteiMi/denseweight 
    # use of package denseweight to calculate weights
    def __init__(self ,alpha, targets_all):
        super(dense_weight_loss_single, self).__init__()
        #put it before as input? as in fit it not in the class?
        self.dw = DenseWeight(alpha=alpha)
        self.dw.fit(targets_all)
    def forward(self, predictions, targets):
        try:

            targs = targets.cpu().detach().numpy()
            weighted_targs = self.dw(targs)
            relevance = torch.from_numpy(weighted_targs).to(torch.device("cuda:0"))
        except ValueError:
            print(
                        'WARNING!)'
                    )
            relevance = torch.ones_like(targets)

        err = torch.pow(predictions - targets, 2)
        err_weighted = relevance * err
        mse = err_weighted.mean()

        return mse


class dense_weight_loss(nn.Module):
    # cite https://github.com/SteiMi/denseweight 
    # use of package denseweight to calculate weights
    def __init__(self ,alpha, targets_all):
        super(dense_weight_loss, self).__init__()
        #self.dw = DenseWeight(alpha=alpha)
        #put it before as input? as in fit it not in the class?
        self.dw_all ={}
        #self.dw_test = DenseWeight(alpha=alpha)
        #testing = self.dw_test.fit(targets_all)
        print(f'Alpha is set to {alpha}!')
        #fit for each label 
        #alphas = [0, 0, 1, 0, 0, 0, 0, 1, 2]
        for i in range(targets_all.shape[1]):
            self.dw_cur = f'dw{i}'
            self.dw_all[self.dw_cur] = DenseWeight(alpha=alpha)
            self.dw_all[self.dw_cur].fit(targets_all[:, i])
        #self.testing = DenseWeight(alpha=alpha)
        #self.testing.fit(targets_all[:,6])
        #self.dw.fit(targets_all)
    def forward(self, predictions, targets):
        try:
            
            relevance = []
            for i in range(predictions.shape[1]):
                self.dw_cur = f'dw{i}'
                targs = targets[:,i].cpu().detach().numpy()
                weighted_targs = self.dw_all[self.dw_cur](targs)
                #relevance[:, i] = torch.from_numpy(weighted_targs).to(torch.device("cuda:0"))
                relevance.append(torch.from_numpy(weighted_targs).to(torch.device("cuda:0")))
            #targs = targets.cpu().detach().numpy()
            #weighted_targs = self.dw(targs)
            #relevance = torch.from_numpy(weighted_targs).to(torch.device("cuda:0"))
        except ValueError:
            print('ValueError')
            relevance = torch.ones_like(targets)
        relevance = torch.stack(relevance, dim = 1)
        err = torch.pow(predictions - targets, 2)
        err_weighted = relevance * err
        mse = err_weighted.mean()

        return mse
    

class dense_weight_loss_tuned(nn.Module):
    # cite https://github.com/SteiMi/denseweight 
    # use of package denseweight to calculate weights
    def __init__(self ,alpha, targets_all):
        super(dense_weight_loss_tuned, self).__init__()
        #self.dw = DenseWeight(alpha=alpha)
        #put it before as input? as in fit it not in the class?
        self.dw_all ={}
        #self.dw_test = DenseWeight(alpha=alpha)
        #testing = self.dw_test.fit(targets_all)
        print(f'Alpha is set to {alpha}!')
        #fit for each label Joy 2 is best bis jzt
        #best [0, 0.1, 0.1, 0.1, 0.2, 2.2, 2,1.2, 0.4, 0.4] 
        #joy was better 2.2
        ###
        ## if nostalgia or tenderness not better, use false
        # made tenderness higher to 0.6
        ## power 7.2
        # joy lower
        # tension lower
        # sadness lower
        alphas = [0, 2, 2, 0.6, 0, 2.2, 1.4 ,1.2, 0.6, 0.1] #statt 3 0.5
        for i in range(targets_all.shape[1]):
            self.dw_cur = f'dw{i}'
            self.dw_all[self.dw_cur] = DenseWeight(alpha=alphas[i])
            self.dw_all[self.dw_cur].fit(targets_all[:, i])
        #self.testing = DenseWeight(alpha=alpha)
        #self.testing.fit(targets_all[:,6])
        #self.dw.fit(targets_all)
        #['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
        self.dense_weight_yes = [False, True, False, True, False, True, True, True, True]
    def forward(self, predictions, targets):
        try:
            
            relevance = []
            for i in range(predictions.shape[1]):
                self.dw_cur = f'dw{i}'
                targs = targets[:,i].cpu().detach().numpy()
                if self.dense_weight_yes[i] == True:
                    weighted_targs = self.dw_all[self.dw_cur](targs)
                    #relevance[:, i] = torch.from_numpy(weighted_targs).to(torch.device("cuda:0"))
                    relevance.append(torch.from_numpy(weighted_targs).to(torch.device("cuda:0")))
                else:
                    weighted_targs = np.ones_like(self.dw_all[self.dw_cur](targs))
                    relevance.append(torch.from_numpy(weighted_targs).to(torch.device("cuda:0")))
            #targs = targets.cpu().detach().numpy()
            #weighted_targs = self.dw(targs)
            #relevance = torch.from_numpy(weighted_targs).to(torch.device("cuda:0"))
        except ValueError:
            print('ValueError')
            relevance = torch.ones_like(targets)
        relevance = torch.stack(relevance, dim = 1)
        err = torch.pow(predictions - targets, 2)
        err_weighted = relevance * err
        mse = err_weighted.mean()

        return mse
        

class adaptive_dense_weight_loss(nn.Module):
    def __init__(self ,alpha, targets_all):
        super(dense_weight_loss_single, self).__init__()
        for i in range(targets_all.shape[1]):
            self.dw = f'dw{i}'
        targets_all[:,0]


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))
    

class dense_weight_loss_RMSE(nn.Module):
    # cite https://github.com/SteiMi/denseweight 
    # use of package denseweight to calculate weights
    def __init__(self ,alpha, targets_all):
        super(dense_weight_loss, self).__init__()
        #self.dw = DenseWeight(alpha=alpha)
        #put it before as input? as in fit it not in the class?
        self.dw_all ={}
        #self.dw_test = DenseWeight(alpha=alpha)
        #testing = self.dw_test.fit(targets_all)
        print(f'Alpha is set to {alpha}!')
        #fit for each label 
        #alphas = [0, 0, 1, 0, 0, 0, 0, 1, 2]
        for i in range(targets_all.shape[1]):
            self.dw_cur = f'dw{i}'
            self.dw_all[self.dw_cur] = DenseWeight(alpha=alpha)
            self.dw_all[self.dw_cur].fit(targets_all[:, i])
        #self.testing = DenseWeight(alpha=alpha)
        #self.testing.fit(targets_all[:,6])
        #self.dw.fit(targets_all)
    def forward(self, predictions, targets):
        try:
            
            relevance = []
            for i in range(predictions.shape[1]):
                self.dw_cur = f'dw{i}'
                targs = targets[:,i].cpu().detach().numpy()
                weighted_targs = self.dw_all[self.dw_cur](targs)
                #relevance[:, i] = torch.from_numpy(weighted_targs).to(torch.device("cuda:0"))
                relevance.append(torch.from_numpy(weighted_targs).to(torch.device("cuda:0")))
            #targs = targets.cpu().detach().numpy()
            #weighted_targs = self.dw(targs)
            #relevance = torch.from_numpy(weighted_targs).to(torch.device("cuda:0"))
        except ValueError:
            print('ValueError')
            relevance = torch.ones_like(targets)
        relevance = torch.stack(relevance, dim = 1)
        err = torch.pow(predictions - targets, 2)
        err_weighted = relevance * err
        mse = err_weighted.mean()

        return torch.sqrt(mse)
    

class dense_weight_loss_tuned_RMSE(nn.Module):
    # cite https://github.com/SteiMi/denseweight 
    # use of package denseweight to calculate weights
    def __init__(self ,alpha, targets_all):
        super(dense_weight_loss_tuned, self).__init__()
        #self.dw = DenseWeight(alpha=alpha)
        #put it before as input? as in fit it not in the class?
        self.dw_all ={}
        #self.dw_test = DenseWeight(alpha=alpha)
        #testing = self.dw_test.fit(targets_all)
        print(f'Alpha is set to {alpha}!')
        #fit for each label Joy 2 is best bis jzt
        #best [0, 0.1, 0.1, 0.1, 0.2, 2.2, 2,1.2, 0.4, 0.4] 
        #joy was better 2.2
        ###
        ## if nostalgia or tenderness not better, use false
        # made tenderness higher to 0.6
        ## power 7.2
        # joy lower
        # tension lower
        # sadness lower
        alphas = [0, 2, 2, 0.6, 0, 2.2, 1.4 ,1.2, 0.6, 0.1] #statt 3 0.5
        for i in range(targets_all.shape[1]):
            self.dw_cur = f'dw{i}'
            self.dw_all[self.dw_cur] = DenseWeight(alpha=alphas[i])
            self.dw_all[self.dw_cur].fit(targets_all[:, i])
        #self.testing = DenseWeight(alpha=alpha)
        #self.testing.fit(targets_all[:,6])
        #self.dw.fit(targets_all)
        #['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']
        self.dense_weight_yes = [False, True, False, True, False, True, True, True, True]
    def forward(self, predictions, targets):
        try:
            
            relevance = []
            for i in range(predictions.shape[1]):
                self.dw_cur = f'dw{i}'
                targs = targets[:,i].cpu().detach().numpy()
                if self.dense_weight_yes[i] == True:
                    weighted_targs = self.dw_all[self.dw_cur](targs)
                    #relevance[:, i] = torch.from_numpy(weighted_targs).to(torch.device("cuda:0"))
                    relevance.append(torch.from_numpy(weighted_targs).to(torch.device("cuda:0")))
                else:
                    weighted_targs = np.ones_like(self.dw_all[self.dw_cur](targs))
                    relevance.append(torch.from_numpy(weighted_targs).to(torch.device("cuda:0")))
            #targs = targets.cpu().detach().numpy()
            #weighted_targs = self.dw(targs)
            #relevance = torch.from_numpy(weighted_targs).to(torch.device("cuda:0"))
        except ValueError:
            print('ValueError')
            relevance = torch.ones_like(targets)
        relevance = torch.stack(relevance, dim = 1)
        err = torch.pow(predictions - targets, 2)
        err_weighted = relevance * err
        mse = err_weighted.mean()

        return torch.sqrt(mse)
    
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))