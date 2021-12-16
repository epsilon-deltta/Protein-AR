import argparse
exam_code = '''
e.g)  
python predict.py -d ./data/sample.txt -m ./models/attns_3.pt
'''
parser = argparse.ArgumentParser("Prediction",epilog=exam_code)   


# parser.add_argument('-s'  ,'--seq'   ,default=None ,metavar='{...}'    ,help='')
parser.add_argument('-d','--dataset'      ,default='./data/sample.txt'  , help='path for dataset')
parser.add_argument('-m','--model'      ,default='./models/attns_3.pt'  , help='model for prediction')
parser.add_argument('-o','--output'      ,default=None  , help='path to save the reuslt')
args = parser.parse_args()


# preprocessing

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

def get_seqStr(path = './data/sample.txt'):
    with open(path,'r',encoding='utf8') as f:
        x_list = f.readlines()

    x_list = list(map(lambda x:x.replace('\n',''),x_list ) )
    return x_list

def seq2int(seq):
    seq2int = [ vocab_map[x] for x in seq ]
    seq2int = torch.tensor(seq2int)
    # seq_tensor = torch.stack(seq)
    return seq2int

def seq2oneHot(seq):
    seq2int = [ vocab_map[x] for x in seq ]
    seq2int = torch.tensor(seq2int)
    oneHot = F.one_hot(seq2int,num_classes=len(vocab_map) )
    return oneHot

def seq_list2int(seq_list):
    tensor_list = list(map(seq2int,seq_list))
    seq_tensor = torch.stack(tensor_list)
    return seq_tensor
def seq_list2oneHot(seq_list):
    tensor_list = list(map(seq2oneHot,seq_list))
    seq_tensor = torch.stack(tensor_list)
    return seq_tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model 
import os
from models import * 
m_path     = args.model
model_name = os.path.basename(m_path).split('_')[0]
model_name = model_name.lower()

model,config = get_model(model_name)
model = model.to(device)

seq_list = get_seqStr(path=args.dataset)

if config['transform'] == 'onehot':
    x = seq_list2oneHot(seq_list).to(device)
elif config['transform'] is None:
    x = seq_list2int(seq_list).to(device)

model.load_state_dict(torch.load(m_path))

# predict 

y = model(x)
y = y.argmax(1)
y = y.cpu().numpy().tolist()

# save output
y = list(map(str,y))
if args.output is None:
    path    = args.dataset
    dirname = os.path.dirname(path)
    fname   = os.path.basename(path).split('.')[0]
    savep   = os.path.join(dirname,f'{fname}_out.csv')
else:
    savep = args.output
print(y)
with open(savep,'w') as f:
    f.write('\n'.join(y) )