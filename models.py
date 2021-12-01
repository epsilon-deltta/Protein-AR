
import torch
# resnet
import torchvision

### x: b,10,20
from torch import nn
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        in_nodes = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_nodes,2)
    def forward(self,x):
        x = x.to(torch.float)
        x = torch.stack([x,x,x],dim=1)
        x = self.backbone(x)
        return x
        
# resnext
import torchvision

### x: b,10,20
from torch import nn
class ResNext(nn.Module):
    def __init__(self):
        super(ResNext,self).__init__()
        self.backbone = torchvision.models.resnext50_32x4d(pretrained=True)
        in_nodes = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_nodes,2)
    def forward(self,x):
        x = x.to(torch.float)
        x = torch.stack([x,x,x],dim=1)
        x = self.backbone(x)
        return x
        
# maxfilterCNN
class squeeze(nn.Module):
    def __init__(self):
        super(squeeze,self).__init__()
    def forward(self,x):
        x = x.squeeze(-1)
        return x
### x: b,10,20
from torch import nn
class MaxFilterCNN(nn.Module):
    def __init__(self):
        super(MaxFilterCNN,self).__init__()
        self.maxFconv = nn.Sequential(
            nn.Conv2d(1,8,kernel_size=(3,20) ),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        
        self.sq = squeeze()
        self.conv1d0 = nn.Sequential(
                    nn.Conv1d(8,8,kernel_size=3),
                    # nn.MaxPool1d(2),
                    nn.BatchNorm1d(8),
                    nn.ReLU()
                )
        self.conv1d1 = nn.Sequential(
                    nn.Conv1d(8,8,kernel_size=2),
                    # nn.MaxPool1d(2),
                    nn.BatchNorm1d(8),
                    nn.ReLU()
                )

        mli = nn.ModuleList([self.maxFconv,self.sq,self.conv1d0,self.conv1d1])
        sample = torch.rand(1,1,10,20)
        for f in mli:
            sample = f(sample)
        b,c,l = sample.shape
        num_node  = c*l 

        self.last = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(num_node,2)
                )
    def forward(self,x):
        x = x.to(torch.float)
        x = x.unsqueeze(1)
        x = self.maxFconv(x)
        x = self.sq(x)
        x = self.conv1d0(x)
        x = self.conv1d1(x)
        x = self.last(x)
        return x
        
# lstm

import torchvision

### x: b,10,20
from torch import nn
class lstm(nn.Module):
    def __init__(self):
        super(lstm,self).__init__()
        self.lstm0 = nn.LSTM(input_size = 20, hidden_size = 20,num_layers=1, batch_first=True)
        self.lstm1 = nn.LSTM(input_size = 20, hidden_size = 20,num_layers=1, batch_first=True)
        self.fc = nn.Linear(20,2)
        
    def forward(self,x):
        x        = x.to(torch.float)
        x,(h,c)  = self.lstm0(x)
        al,(x,c) = self.lstm1(x)
        x        = x.transpose(0,1)
        x        = x.squeeze()
        x        = self.fc(x)
        return x

    
# lstm0: /wo oneHot encoding
import torchvision

### x: b,10,20
from torch import nn
class lstm0(nn.Module):
    def __init__(self):
        super(lstm0,self).__init__()
        self.lstm0 = nn.LSTM(input_size = 1, hidden_size = 10,num_layers=1, batch_first=True)
        self.lstm1 = nn.LSTM(input_size = 10, hidden_size = 10,num_layers=1, batch_first=True)
        self.fc = nn.Linear(10,2)
        
    def forward(self,x):
        x        = x.to(torch.float)
        
        x        = x.unsqueeze(-1)
        x,(h,c)  = self.lstm0(x)
        al,(x,c) = self.lstm1(x)
        x        = x.transpose(0,1)
        x        = x.squeeze()
        x        = self.fc(x)
        return x
# lstm1
# lstm: /w oneHot encoding

### x: b,10,20
from torch import nn
class lstm1(nn.Module):
    def __init__(self):
        super(lstm1,self).__init__()
        self.emb   = nn.Embedding(20,20)
        self.lstm0 = nn.LSTM(input_size = 20, hidden_size = 20,num_layers=1, batch_first=True)
        self.lstm1 = nn.LSTM(input_size = 20, hidden_size = 20,num_layers=1, batch_first=True)
        self.fc = nn.Linear(20,2)
        
    def forward(self,x):
        x        = self.emb(x)
        x,(h,c)  = self.lstm0(x)
        al,(x,c) = self.lstm1(x)
        x        = x.transpose(0,1)
        x        = x.squeeze()
        x        = self.fc(x)
        return x
        
# self-attn

import torchvision

### x: b,10,20
from torch import nn
class attns(nn.Module):
    def __init__(self):
        super(attns,self).__init__()
        self.attn0 = nn.TransformerEncoderLayer(d_model=20, nhead=4,batch_first=True)
        self.attn1 = nn.TransformerEncoderLayer(d_model=20, nhead=4,batch_first=True)
        self.flat  = nn.Flatten()
        self.fc    = nn.Sequential(
            nn.Linear(200,100),
            nn.ReLU(),
            nn.Linear(100,2)
        )
        
    def forward(self,x):
        x = x.to(torch.float)
        x = self.attn0(x)+x
        x = self.attn1(x)+x
        x = self.flat(x)
        x = self.fc(x)
        return x
        
import torchvision

### x: b,10,20
from torch import nn
class attns0(nn.Module):
    def __init__(self):
        super(attns0,self).__init__()
        self.attn0 = nn.TransformerEncoderLayer(d_model=1, nhead=1,batch_first=True)
        self.attn1 = nn.TransformerEncoderLayer(d_model=1, nhead=1,batch_first=True)
        self.flat  = nn.Flatten()
        self.fc    = nn.Sequential(
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,2)
        )
        
    def forward(self,x):
        x = x.to(torch.float) # b,10
        x = x.unsqueeze(-1)   # b,10,1
        x = self.attn0(x)+x   # b,10,1
        x = self.attn1(x)+x   # b,10,1
        x = self.flat(x)      # b,10
        x = self.fc(x)        # b,2
        return x
        
import torchvision

# attns1
### x: b,10,20
from torch import nn
class attns1(nn.Module):
    def __init__(self):
        super(attns1,self).__init__()
        self.emb   = nn.Embedding(20,20)
        self.attn0 = nn.TransformerEncoderLayer(d_model=20, nhead=4,batch_first=True)
        self.attn1 = nn.TransformerEncoderLayer(d_model=20, nhead=4,batch_first=True)
        self.flat  = nn.Flatten()
        self.fc    = nn.Sequential(
            nn.Linear(200,100),
            nn.ReLU(),
            nn.Linear(100,2)
        )
        
    def forward(self,x):

        x = self.emb(x)       # b,10,20
        x = self.attn0(x)+x   # b,10,20
        x = self.attn1(x)+x   # b,10,20
        x = self.flat(x)      # b,200
        x = self.fc(x)        # b,2
        return x
        

def get_model(model:str= 'attn'):
    model = model.lower()
    transform ='onehot'
    if model.startswith('maxfil'):
        model = MaxFilterCNN()
        transform = 'onehot'
    elif model.startswith('resnet'):
        model = ResNet()
        transform = 'onehot'
    elif model.startswith('resnext'):
        model = ResNext()
        transform = 'onehot'
        
    elif model.startswith('attn'):
        if (model == 'attn') | (model == 'attns') :
            transform = 'onehot'
            model = attns()
        elif model == 'attns0':
            transform = None
            model = attns0()
        elif model == 'attns1':
            transform = None
            model = attns1()
    elif model.startswith('lstm'):
        if model == 'lstm':
            model = lstm()
            transform = 'onehot'
        elif model == 'lstm0':
            model = lstm0()
            transform = None
        elif model == 'lstm1':
            model = lstm1()
            transform = None
    else:
        raise ValueError(f"There is no '{model}' ")
    return model,transform