import torch
import torch.nn.functional as F
import torch.nn as nn


class gcn(torch.nn.Module):
    def __init__(self, k, d):
        super(gcn,self).__init__()
        D=k*d
        self.fc = torch.nn.Linear(2*D, D)
        self.dropout = nn.Dropout(p=0.1)
    def forward(self,X, STE, A):
        X = torch.cat((X, STE), dim=-1)
        H = F.gelu(self.fc(X))  # [batch_size, num_steps, num_nodes, K * d]
        H = torch.einsum('ncvl,vw->ncwl',(H,A))    #[batch_size, num_steps, num_nodes, D]  [N,N]  [batch_size, num_steps, num_nodes, D]

        return self.dropout(H.contiguous())

class randomGAT(torch.nn.Module):
    def __init__(self, k, d, adj,device):
        super(randomGAT, self).__init__()
        D=k*d
        self.d = d
        self.K = k
        num_nodes=adj.shape[0]
        self.device=device
        self.fc = torch.nn.Linear(2*D, D)
        self.adj=adj
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)


    def forward(self,X, STE):
        X = torch.cat((X, STE), dim=-1)
        H = F.gelu(self.fc(X))  # [batch_size, num_steps, num_nodes, K * d]
        H = torch.cat(torch.split(H, self.d, dim=-1), dim=0)  # [k* batch_size, num_steps, num_nodes, d]
        adp =torch.mm(self.nodevec1, self.nodevec2)
        zero_vec = torch.tensor(-9e15).to(self.device)
        adp= torch.where(self.adj > 0, adp, zero_vec)
        adj = F.softmax(adp, dim=-1)
        H = torch.einsum('vw,ncwl->ncvl',(adj,H))    #[batch_size, num_steps, num_nodes, D]  [N,N]  [batch_size, num_steps, num_nodes, D]
        #H = torch.matmul(adj, H)
        H = torch.cat(torch.split(H, H.shape[0] // self.K, dim=0), dim=-1)
        return F.gelu(H.contiguous())


class STEmbModel(torch.nn.Module):
    def __init__(self, SEDims, TEDims, OutDims, device):
        super(STEmbModel, self).__init__()
        self.TEDims = TEDims
        self.fc3 = torch.nn.Linear(SEDims, OutDims)
        self.fc4 = torch.nn.Linear(OutDims, OutDims)
        self.fc5 = torch.nn.Linear(TEDims, OutDims)
        self.fc6 = torch.nn.Linear(OutDims, OutDims)
        self.device = device


    def forward(self, SE, TE):
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.fc4(F.gelu(self.fc3(SE)))
        dayofweek = F.one_hot(TE[..., 0], num_classes = 7)
        timeofday = F.one_hot(TE[..., 1], num_classes = self.TEDims-7)
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        TE = TE.unsqueeze(2).type(torch.FloatTensor).to(self.device)
        TE = self.fc6(F.gelu(self.fc5(TE)))
        sum_tensor = torch.add(SE, TE)
        return sum_tensor


class SpatialAttentionModel(torch.nn.Module):
    def __init__(self, K, d, adj, dropout=0.3,mask=True):
        super(SpatialAttentionModel, self).__init__()
        '''
        spatial attention mechanism
        X:      [batch_size, num_step, N, D]
        STE:    [batch_size, num_step, N, D]
        K:      number of attention heads
        d:      dimension of each attention outputs
        return: [batch_size, num_step, N, D]
        '''
        D = K*d
        self.fc7 = torch.nn.Linear(2*D, D)
        self.fc8 = torch.nn.Linear(2*D, D)
        self.fc9 = torch.nn.Linear(2*D, D)
        self.fc10 = torch.nn.Linear(D, D)
        self.fc11 = torch.nn.Linear(D, D)
        self.K = K
        self.d = d
        self.adj = adj
        self.mask = mask
        self.dropout=dropout
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, X, STE):
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_steps, num_nodes, 2D]

        query = F.gelu(self.fc7(X))   # [batch_size, num_steps, num_nodes, K * d]
        key = F.gelu(self.fc8(X))
        value = F.gelu(self.fc9(X))

        # [K * batch_size, num_steps, num_nodes, d]
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)

        # [K*batch_size, num_steps, num_nodes, num_nodes]=[K * batch_size, num_steps, num_nodes, d] @ [K * batch_size, num_steps, d, num_nodes]
        attention = torch.matmul(query, torch.transpose(key, 2, 3)) #[128,12,207,8] @ [128,12,8,207]=[128,12,207,207]
        attention /= (self.d ** 0.5)
        if self.mask:
            zero_vec = -9e15 * torch.ones_like(attention)              #根据attention设立mask
            attention = torch.where(self.adj > 0, attention, zero_vec) #如果adj位置元素大于0，则保留attention系数，不然设置为一个很大的数(sfoemax下可以当做0)
        attention = self.softmax(attention)
        #attention = F.dropout(attention, self.dropout, training=self.training)

        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, X.shape[0]//self.K, dim=0), dim=-1)
        X = self.fc11(F.gelu(self.fc10(X)))

        #X = F.relu(self.fc10(X))
        #X=  self.fc11(F.dropout(X, self.dropout, training=self.training))

        return X


class TemporalAttentionModel(torch.nn.Module):
    def __init__(self, K, d, device):
        '''
        spatial attention mechanism
        X:      [batch_size, num_step, N, D]
        STE:    [batch_size, num_step, N, D]
        K:      number of attention heads
        d:      dimension of each attention outputs
        return: [batch_size, num_step, N, D]
        '''
        super(TemporalAttentionModel, self).__init__()
        D = K*d
        self.fc12 = torch.nn.Linear(2*D, D)
        self.fc13 = torch.nn.Linear(2*D, D)
        self.fc14 = torch.nn.Linear(2*D, D)
        self.fc15 = torch.nn.Linear(D, D)
        self.fc16 = torch.nn.Linear(D, D)
        self.K = K
        self.d = d
        self.device =device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X, STE, Mask=True):
        X = torch.cat((X, STE), dim=-1)   # [batch_size, num_steps, num_nodes, 2D]
        query = F.gelu(self.fc12(X))  # [batch_size, num_steps, num_nodes, K * d]
        key = F.gelu(self.fc13(X))
        value = F.gelu(self.fc14(X))
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        # [K * batch_size, num_steps, num_nodes, d]

        # query: [K * batch_size, num_nodes, num_step, d]
        # key:   [K * batch_size, num_nodes, d, num_step]
        # value: [K * batch_size, num_nodes, num_step, d]
        query = torch.transpose(query, 2, 1)
        key = torch.transpose(torch.transpose(key, 1, 2), 2, 3)
        #[K * batch_size, num_steps, num_nodes, d] - [K * batch_size, num_nodes, num_steps, d] - [K * batch_size, num_nodes, d, num_steps]
        value = torch.transpose(value, 2, 1)

        # [K * batch_size, N, num_step, num_step] = [K * batch_size, num_nodes, num_step, d] @ [K * batch_size, num_nodes, d, num_steps]
        attention = torch.matmul(query, key)  #[128,12,207,8] @ [128,12,8,207]=[128,12,207,207]
        attention /= (self.d ** 0.5)
        # attention = F.dropout(attention, self.dropout, training=self.training)

        if Mask == True:
            batch_size = X.shape[0]
            num_steps = X.shape[1]
            num_vertexs = X.shape[2]
            mask = torch.ones(num_steps, num_steps).to(self.device) # [T,T]
            mask = torch.tril(mask) # [T,T]下三角为1其余为0
            zero_vec = torch.tensor(-9e15).to(self.device)
            mask = mask.to(torch.bool)  #里面元素全是负无穷大
            attention = torch.where(mask, attention, zero_vec)

        attention = self.softmax(attention)
        #X = torch.einsum('nclv,ncvw->nclw', (attention, value))  #两个应该是等价的
        X = torch.matmul(attention, value)
        X = torch.transpose(X, 2, 1)
        X = torch.cat(torch.split(X, X.shape[0]//self.K, dim=0), dim=-1)
        X = self.dropout(self.fc16(F.gelu(self.fc15(X))))

        #X = F.relu(self.fc15(X))
        #X=  self.fc16(F.dropout(X, self.dropout, training=self.training))
        return X



class GatedFusionModel(torch.nn.Module):
    def __init__(self, K, d):
        super(GatedFusionModel, self).__init__()
        D = K*d
        self.fc17 = torch.nn.Linear(D, D)
        self.fc18 = torch.nn.Linear(D, D)
        self.fc19 = torch.nn.Linear(D, D)
        self.fc20 = torch.nn.Linear(D, D)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, HS, HT):
        XS = self.fc17(HS)
        XT = self.fc18(HT)
        z = self.sigmoid(torch.add(XS, XT))
        H = torch.add((z* HS), ((1-z)* HT))
        H = self.fc20(F.gelu(self.fc19(H)))
        return H


class STAttModel(torch.nn.Module):
    def __init__(self, K, d,adj, device):
        super(STAttModel, self).__init__()
        D = K * d
        self.fc30 = torch.nn.Linear(7*D, D)
        self.gcn = gcn(K, d)
        self.gcn1 = randomGAT(K, d, adj[0], device)
        self.gcn2 = randomGAT(K, d, adj[0], device)
        self.gcn3 = randomGAT(K, d, adj[1], device)
        self.gcn4 = randomGAT(K, d, adj[1], device)
        self.temporalAttention = TemporalAttentionModel(K, d, device)
        self.gatedFusion = GatedFusionModel(K, d)

    def forward(self, X, STE,adp, Mask=True):
        HS1 = self.gcn1(X, STE)
        HS2 = self.gcn2(HS1, STE)
        HS3 = self.gcn3(X, STE)
        HS4 = self.gcn4(HS3, STE)
        HS5 = self.gcn(X, STE, adp)
        HS6 = self.gcn(HS5, STE, adp)
        HS = torch.cat((X,HS1,HS2,HS3,HS4,HS5,HS6), dim=-1)
        HS = F.gelu(self.fc30(HS))
        HT = self.temporalAttention(X, STE, Mask)
        H = self.gatedFusion(HS, HT)
        return torch.add(X, H)


class TransformAttentionModel(torch.nn.Module):
    def __init__(self, K, d, dropout=0.3):
        super(TransformAttentionModel, self).__init__()
        D = K * d
        self.fc21 = torch.nn.Linear(D, D)
        self.fc22 = torch.nn.Linear(D, D)
        self.fc23 = torch.nn.Linear(D, D)
        self.fc24 = torch.nn.Linear(D, D)
        self.fc25 = torch.nn.Linear(D, D)
        self.K = K
        self.d = d
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = dropout

    def forward(self, X, STE_P, STE_Q ,mask1 =False):
        query = F.gelu(self.fc21(STE_Q))
        key = F.gelu(self.fc22(STE_P))
        value = F.gelu(self.fc23(X))
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        query = torch.transpose(query, 2, 1)                            # [K * batch_size, num_nodes, num_steps, d]
        key = torch.transpose(torch.transpose(key, 1, 2), 2, 3)         # [K * batch_size, num_nodes, d, num_steps]
        value = torch.transpose(value, 2, 1)

        attention = torch.matmul(query, key)                            # [K * batch_size, num_nodes, num_steps, num_steps]
        attention /= (self.d ** 0.5)
        attention = self.softmax(attention)
        '''
        if mask1:
            num_step = X.shape[1]
            num_nodes = X.shape[2]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)  # 取下三角矩阵，包括对角线
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)  # 【1，1，num_step, num_step】
            mask = mask.repeat(self.K * batch_size, num_nodes, 1, 1)
            # [K * batch_size, num_nodes, num_steps, num_nodes]
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, -2 ** 15 + 1)
        '''


        X = torch.matmul(attention, value)
        X = torch.transpose(X, 2, 1)
        X = torch.cat(torch.split(X, X.shape[0]//self.K, dim=0), dim=-1)
        X = self.fc25(F.gelu(self.fc24(X)))

        #X = F.relu(self.fc24(X))
        #X=  self.fc25(F.dropout(X, self.dropout, training=self.training))
        return X


class RGDAN(torch.nn.Module):
    def __init__(self, K, d, SEDims, TEDims, P, L, device, adj,num_nodes):
        super(RGDAN, self).__init__()
        D = K*d
        self.fc1 = torch.nn.Linear(1, D)
        self.fc2 = torch.nn.Linear(D, D)
        self.STEmb = STEmbModel(SEDims, TEDims, K*d, device)
        self.STAttBlockEnc = STAttModel(K, d, adj, device)
        self.STAttBlockDec = STAttModel(K, d, adj, device)
        self.transformAttention = TransformAttentionModel(K, d)
        self.P = P
        self.L = L
        self.device = device
        self.fc26 = torch.nn.Linear(D, D)
        self.fc27 = torch.nn.Linear(D, 1)
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)

        self.dropout = nn.Dropout(p=0.1)
    def forward(self, X, SE, TE):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)#这里是二维，所以可以用1，如果是三维或者四维，那么dim=-1会比较好

        '''
        adp = (torch.mm(self.nodevec1, self.nodevec2))
        adp =torch.mm(self.nodevec1, self.nodevec2)
        zero_vec = torch.tensor(-9e15).to(self.device)
        #adp = torch.tanh(adp)    #貌似不加也没事
        adp= torch.where(adp > 0, adp, zero_vec)
        adp = F.softmax(adp, dim=-1)        
        '''

        X = X.unsqueeze(3)
        X = self.fc2(F.gelu(self.fc1(X)))
        STE = self.STEmb(SE, TE)
        STE_P = STE[:, : self.P]
        STE_Q = STE[:, self.P :]
        X = self.STAttBlockEnc(X, STE_P, adp, Mask=True)
        X = self.transformAttention(X, STE_P, STE_Q)
        X = self.STAttBlockDec(X, STE_Q, adp, Mask=True)
        X = self.fc27(self.dropout(F.gelu(self.fc26(X))))
        #X = F.relu(self.fc26(X))
        #X = self.fc27(F.dropout(X, self.dropout, training=self.training))
        return X.squeeze(3)


def mae_loss(pred, label, device):
    mask = (label != 0)
    mask = mask.type(torch.FloatTensor).to(device)
    mask /= torch.mean(mask)
    mask[mask!=mask] = 0
    loss = torch.abs(pred - label)
    loss *= mask
    loss[loss!=loss] = 0
    loss = torch.mean(loss)
    return loss

def mse_loss(pred, label, device):
    mask = (label != 0)
    mask = mask.type(torch.FloatTensor).to(device)
    mask /= torch.mean(mask)
    mask[mask!=mask] = 0
    loss = (pred-label)**2
    loss *= mask
    loss[loss!=loss] = 0
    loss = torch.mean(loss)
    return loss

def mape_loss(pred, label, device):
    mask = (label != 0)
    mask = mask.type(torch.FloatTensor).to(device)
    mask /= torch.mean(mask)
    mask[mask!=mask] = 0
    loss = torch.abs(pred - label)/label
    loss *= mask
    loss[loss!=loss] = 0
    loss = torch.mean(loss)
    return loss