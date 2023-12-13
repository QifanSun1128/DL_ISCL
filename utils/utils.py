import os
import torch
import torch.nn as nn
import shutil


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


def save_checkpoint(
    state, is_best, checkpoint="checkpoint", filename="checkpoint.pth.tar"
):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))

def get_centers(net, dataloader, num_classes):        
    centers = 0 
    refs = torch.LongTensor(range(num_classes)).unsqueeze(1)
    if torch.cuda.is_available():
        ref = refs.cuda()
        
    for sample in iter(dataloader):
        data = sample[0]
        gt = sample[1]
        if torch.cuda.is_available():
            data = data.cuda()    
            gt = gt.cuda()   
        batch_size = data.size(0)

        output = net.forward(data)
        feature = output.data 
        feat_len = feature.size(1)
    
        gt = gt.unsqueeze(0).expand(num_classes, -1)
        mask = (gt == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
        feature = feature.unsqueeze(0)
        # update centers
        centers += torch.sum(feature * mask, dim=1)

    return centers