import argparse
exam_code = '''
e.g)  
python evaluate.py -d ./models
'''
parser = argparse.ArgumentParser("Evaluate models",epilog=exam_code)   


parser.add_argument('-d'  ,'--directory'   ,default='models' ,metavar='{...}'    ,help='directory path containing the models')
# parser.add_argument('-p'  ,'--path'      ,default=None       , help='Specify the model')
parser.add_argument('--dataset_path'      ,default='./data/split/test.csv'  , help='test dataset path')
args = parser.parse_args()



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
from torch import nn
from torch.nn import functional as F
import pandas as pd
class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self,path):
        self.df = pd.read_csv(path)

    def __getitem__(self,idx):
        item = self.df.iloc[idx]
        x = item['seq']
        y = item['label']
        
        x = self.seq2oneHot(x)
        y = torch.tensor(y)
        # y = self.label2oneHot(y)
        return x,y
    
    def __len__(self):
        return len(self.df)
    
    def seq2oneHot(self,seq):
        seq2int = [ vocab_map[x] for x in list(seq) ]
        seq2int = torch.tensor(seq2int)
        oneHot = F.one_hot(seq2int,num_classes=len(vocab_map) )
        return oneHot
    
    def label2oneHot(self,label):
        return F.one_hot(torch.tensor(label),num_classes=2)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tedt  = ProteinDataset(args.dataset_path)
tedl  = torch.utils.data.DataLoader(tedt, batch_size=64, num_workers=4)

# setting

loss = nn.CrossEntropyLoss()

import os
files = os.listdir(args.directory)

model_paths = [os.path.join('./models',file) for file in files if file.endswith('.pt')]

import os
from models import * 
results = {}

# evaluate 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import precision_score 
from sklearn.metrics import f1_score 
# from sklearn.metrics import make_scorer
# from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 

def evaluate(dl,model,lossf,epoch=None):
    model.eval()
    size, _ , losses = len(dl.dataset) ,0,0
    pre_l,gt_l = [],[]
    with torch.no_grad():
        for x,y in dl:
            x,y = x.to(device),y.to(device)
            pre = model(x)
            loss = lossf(pre,y)
            
            losses += loss.item()
            pre_l.extend(pre.argmax(1).cpu().numpy().tolist())
            gt_l .extend(y.cpu().numpy().tolist())
    
    loss     = losses/size
    acc      = accuracy_score(gt_l,pre_l)
    recall   = recall_score(gt_l,pre_l)
    precision= precision_score(gt_l,pre_l)
    f1       = f1_score(gt_l,pre_l)
    confusion= confusion_matrix(gt_l,pre_l)

    metrics = {'acc':acc,'recall':recall,'precision':precision,'f1':f1,'confusion':confusion,'loss':loss}
    return metrics

for m_path in model_paths:
    model_name = os.path.basename(m_path).split('_')[0]
    model_name = model_name.lower()
    
    model = model_name

    model = model.lower()
    if model.startswith('maxfil') :
        model = MaxFilterCNN().to(device)
    elif model.startswith('resnet'):
        model = ResNet().to(device)
    elif model.startswith('resnext'):
        model = ResNext().to(device)
    elif model.startswith('attn'):
        model = attns().to(device)
    elif model.startswith('lstm'):
        model = lstm().to(device)
    
    
    model.load_state_dict(torch.load(m_path))
    
    result = evaluate(tedl,model,loss)
    
    print(f'{model_name}: {result}')
    results[model_name] = result


# save the reuslt as csv file

import pandas as pd
df  = pd.DataFrame(results).T
models = [os.path.splitext( os.path.basename(path) )[0] for path in model_paths]
df.to_csv(f"assets/{'&'.join(models)}.csv")
print(f"result was saved in assets/{'&'.join(models)}.csv")