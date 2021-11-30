import argparse
exam_code = '''
e.g)  
python train.py -m attn
'''
parser = argparse.ArgumentParser("Train datasets",epilog=exam_code)   

# parser.add_argument('-d'  ,'--dt'      ,default='pf'      ,metavar='{pf,bln}' , help='Dataset')
parser.add_argument('-m'  ,'--model'   ,default='attns' ,metavar='{...}'    ,help='model name')

args = parser.parse_args()

# model load and settings
from models import *

model = args.model
device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = model.lower()
if model.startswith('maxfil'):
    model = MaxFilterCNN().to(device)
elif model.startswith('resnet'):
    model = ResNet().to(device)
elif model.startswith('resnext'):
    model = ResNext().to(device)
elif model.startswith('attn'):
    if model == 'attn':
        transform = 'onehot'
        model = attns().to(device)
    elif model == 'attn0':
        transform = None
        model = attns0().to(device)
elif model.startswith('lstm'):
    if model == 'lstm':
        model = lstm().to(device)
        transform = 'onehot'
    elif model == 'lstm0':
        model = lstm0().to(device)
        transform = None

loss = nn.CrossEntropyLoss()
params = [p for p in model.parameters() if p.requires_grad]
opt  = torch.optim.Adam(params)

# dataset and loader

def get_vocab_map(vocab_path='./data/vocab.txt'):
    with open(vocab_path,'r') as f:
        vocab = f.read()

    vocab = vocab.replace('\n','')

    import re
    p = re.compile('\s+')
    vocab = re.sub(p,' ',vocab)

    import ast
    vocab = ast.literal_eval(vocab.split('=')[1].strip())

    len_vocab = len(vocab)
    vocab_map = dict(zip(vocab,range(len_vocab)))
    return vocab_map
vocab_map =get_vocab_map()

import torch
from torch.nn import functional as F
import pandas as pd
class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self,path,transform='oneHot'):
        self.df        = pd.read_csv(path)
        if type(transform) == str:
            self.transform = transform.lower()
        else:
            self.transform = transform
        
    def __getitem__(self,idx):
        item = self.df.iloc[idx]
        x = item['seq']
        y = item['label']
        
        if self.transform == 'onehot':
            x = self.seq2oneHot(x)
        elif self.transform is None:
            x = self.seq2int(x)
            
        y = torch.tensor(y)
        # y = self.label2oneHot(y)
        return x,y
    
    def __len__(self):
        return len(self.df)
    
    def seq2int(self,seq):
        seq2int = [ vocab_map[x] for x in list(seq) ]
        seq2int = torch.tensor(seq2int)
        return seq2int
    
    def seq2oneHot(self,seq): # 'ABCDE..' -> OneHot (10,20) 
        seq2int = self.seq2seq_int(seq)
        oneHot = F.one_hot(seq2int,num_classes=len(vocab_map) )
        return oneHot
    
    def label2oneHot(self,label):
        return F.one_hot(torch.tensor(label),num_classes=2)
    
trdt  = ProteinDataset('./data/split/train.csv',transform=transform)
valdt = ProteinDataset('./data/split/val.csv'  ,transform=transform)
tedt  = ProteinDataset('./data/split/test.csv' ,transform=transform)

trdl  = torch.utils.data.DataLoader(trdt, batch_size=64, num_workers=4)
valdl  = torch.utils.data.DataLoader(valdt, batch_size=64, num_workers=4)
tedl  = torch.utils.data.DataLoader(tedt, batch_size=64, num_workers=4)


# train/validate

def train(dl,model,lossf,opt):
    model.train()
    for x,y in dl:
        x,y = x.to(device),y.to(device)
        pre = model(x)
        loss = lossf(pre,y)

        opt.zero_grad()
        loss.backward()
        opt.step()

def test(dl,model,lossf,epoch=None):
    model.eval()
    size, acc , losses = len(dl.dataset) ,0,0
    with torch.no_grad():
        for x,y in dl:
            x,y = x.to(device),y.to(device)
            pre = model(x)
            loss = lossf(pre,y)
    
            acc += (pre.argmax(1)==y).type(torch.float).sum().item()
            losses += loss.item()
    accuracy = round(acc/size,4)
    val_loss = round(losses/size,6)
    print(f'[{epoch}] acc/loss: {accuracy}/{val_loss}')
    return accuracy,val_loss

import copy
patience = 5
val_losses = {0:1}
for i in range(100):
    train(trdl,model,loss,opt)
    acc,val_loss = test(valdl,model,loss,i)
    
    
    if min(val_losses.values() ) > val_loss:
        val_losses[i] = val_loss
        best_model = copy.deepcopy(model)
    if i == min(val_losses,key=val_losses.get)+patience:
        break
        
import os
model_name = f"{best_model.__str__().split('(')[0]}_{max(val_losses)}.pt"
model_path = os.path.join('./models',model_name) 
torch.save(best_model.state_dict(),model_path)