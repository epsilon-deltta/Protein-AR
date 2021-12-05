import argparse
import torch
exam_code = '''
e.g)  
python train.py -m attn
'''
parser = argparse.ArgumentParser("Train datasets",epilog=exam_code)   

# parser.add_argument('-d'  ,'--dt'      ,default='pf'      ,metavar='{pf,bln}' , help='Dataset')
parser.add_argument('-m'  ,'--model'   ,default='attns' ,metavar='{...}'    ,help='model name')
parser.add_argument('--batch_size'   ,default=None,type=int     ,help='batch size')
parser.add_argument('--device'   ,default=None,type=str     ,help='cpu | gpu')
args = parser.parse_args()

args.model = args.model.lower()
if args.device is None:
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'