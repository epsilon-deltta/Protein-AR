import argparse
exam_code = '''
e.g)  
python evaluate.py -d ./models
'''
parser = argparse.ArgumentParser("Evaluate models",epilog=exam_code)   


parser.add_argument('-d'  ,'--directory'   ,default='models' ,metavar='{...}'    ,help='directory path containing the models')
parser.add_argument('-p'  ,'--path'      ,default=None       , help='Specify the model path')
parser.add_argument('--dataset_path'      ,default='./data/split/test.csv'  , help='test dataset path')
parser.add_argument('-s','--save'      ,default=True  , help='whether to save')
args = parser.parse_args()


# dataset class
# model
# evaluate

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
        seq2int = self.seq2int(seq)
        oneHot = F.one_hot(seq2int,num_classes=len(vocab_map) )
        return oneHot
    
    def label2oneHot(self,label):
        return F.one_hot(torch.tensor(label),num_classes=2)
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from models import *
import os
results = {}
model_paths = []
if args.path is not None:
    m_path = args.path
    model_paths.append(m_path)
    model_name = os.path.basename(m_path).split('_')[0].lower()
    print(model_name)
    # model = 'lstm0'
    model,transform = get_model(model_name)
    model = model.to(device)

    
    tedt  = ProteinDataset(args.dataset_path,transform=transform)
    tedl  = torch.utils.data.DataLoader(tedt, batch_size=64, num_workers=4)
    
    loss = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    opt  = torch.optim.Adam(params)

    
    model.load_state_dict(torch.load(m_path))
    
    result = evaluate(tedl,model,loss)
    
    print(f'{model_name}: {result}')
    results[model_name] = result

else:
    files = os.listdir(args.directory)
    model_paths = [os.path.join('./models',file) for file in files if file.endswith('.pt')]
    
    for m_path in model_paths:
        model_name = os.path.basename(m_path).split('_')[0].lower()
        
        print(model_name)
        model,transform = get_model(model_name)
        model = model.to(device)
        
        tedt  = ProteinDataset(args.dataset_path,transform=transform)
        tedl  = torch.utils.data.DataLoader(tedt, batch_size=64, num_workers=4)
        
        loss = nn.CrossEntropyLoss()
        params = [p for p in model.parameters() if p.requires_grad]
        opt  = torch.optim.Adam(params)

        model.load_state_dict(torch.load(m_path))

        result = evaluate(tedl,model,loss)

        print(f'{model_name}: {result}')
        results[model_name] = result
# save the results

if args.save:
    import pandas as pd
    df  = pd.DataFrame(results).T
    models = [os.path.splitext( os.path.basename(path) )[0] for path in model_paths]
    df.to_csv(f"assets/{'&'.join(models)}.csv")
    print(f"result was saved in assets/{'&'.join(models)}.csv")
    