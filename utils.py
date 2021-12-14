from torch import nn
def get_loss(name:str='crossentropy'):
    loss = None
    if name == 'crossentropy':
        loss = nn.CrossEntropyLoss()
    elif name == 'mse':
        loss = nn.MSELoss()
    return loss