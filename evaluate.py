import torch
from dataset import ProteinDataset

from sklearn.metrics import accuracy_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import precision_score 
from sklearn.metrics import f1_score 
# from sklearn.metrics import make_scorer
# from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 

def evaluate(dl,model,lossf,epoch=None,device='cuda'):
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

if __name__ == '__main__':

    from options.eval_op import args

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
        
        model,config = get_model(model_name)
        model = model.to(device)

         # config
        transform = config['transform'] 
        batch_size = config['batch_size']
        
        tedt  = ProteinDataset(args.dataset_path,transform=transform)
        tedl  = torch.utils.data.DataLoader(tedt, batch_size=batch_size, num_workers=4)

        loss = nn.CrossEntropyLoss()
        params = [p for p in model.parameters() if p.requires_grad]
        opt  = torch.optim.Adam(params)


        model.load_state_dict(torch.load(m_path))

        result = evaluate(tedl,model,loss,device=device)

        print(f'{model_name}: {result}')
        results[model_name] = result

    else:
        files = os.listdir(args.directory)
        model_paths = [os.path.join('./models',file) for file in files if file.endswith('.pt')]

        for m_path in model_paths:
            model_name = os.path.basename(m_path).split('_')[0].lower()

            print(model_name)
            model,config = get_model(model_name)
            model = model.to(device)

            # config
            transform = config['transform']
            batch_size = config['batch_size']

            tedt  = ProteinDataset(args.dataset_path,transform=transform)
            tedl  = torch.utils.data.DataLoader(tedt, batch_size=batch_size, num_workers=4)

            loss = nn.CrossEntropyLoss()
            params = [p for p in model.parameters() if p.requires_grad]
            opt  = torch.optim.Adam(params)

            model.load_state_dict(torch.load(m_path))

            result = evaluate(tedl,model,loss,device=device)

            print(f'{model_name}: {result}')
            results[model_name] = result
    # save the results
    print(type(args.save ))
    if args.save:
        import pandas as pd
        df  = pd.DataFrame(results).T
        models = [os.path.splitext( os.path.basename(path) )[0] for path in model_paths]
        df.to_csv(f"assets/{'&'.join(models)}.csv")
        print(f"result was saved in assets/{'&'.join(models)}.csv")
