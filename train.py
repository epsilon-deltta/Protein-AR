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

model = args.model.lower()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model,transform = get_model(model)
model = model.to(device)

loss = nn.CrossEntropyLoss()
params = [p for p in model.parameters() if p.requires_grad]
opt  = torch.optim.Adam(params)

# dataset and loader

from dataset import ProteinDataset
    
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