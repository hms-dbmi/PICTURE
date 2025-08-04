import argparse
import os
import torch
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.optim import Adam, AdamW, RMSprop # optmizers
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau # Learning rate schedulers
from utils.train_utils import cnn_train,test_model,adjust_learning_rate,save_model
from data.brain import BrainHDF5Dataset
from ctran import ctranspath
from collections import OrderedDict

#PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

#PyTorch
ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss


def main(args):
    print("input args:::", args)

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    print("MODEL IS SAVING TO:", args.save_folder)

    device_lst = list(range(args.device))
    device_str = ",".join(str(e) for e in device_lst)
    os.environ['CUDA_VISIBLE_DEVICES'] = device_str
    ##############################

    if args.model == 'CNN':
        model = timm.create_model('resnet18',pretrained=True)
        head = nn.Sequential(nn.Linear(model.fc.in_features, 128),
                             nn.ReLU(), nn.Linear(128,2))
        model.fc = head

    elif args.model == 'SWIN':
        model = timm.create_model("swin_base_patch4_window7_224_in22k", pretrained=True)
        head = nn.Sequential(nn.Linear(model.head.in_features, 128),
                             nn.ReLU(), nn.Linear(128, 2))
        model.head = head

    elif args.model == 'cTrans':
        model = ctranspath()
        model.head = nn.Identity()
        td = torch.load(r'./pretrained/ctranspath.pth')
        model.load_state_dict(td['model'], strict=True)

        head = nn.Sequential(nn.Linear(768, 128),
                            nn.ReLU(), nn.Linear(128, 2))

        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        model.head = head


    


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=device_lst)

    if args.onlytest:

        my_dict = torch.load(args.test_model_path)['model']
        new_state_dict = OrderedDict()
        for k, v in my_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)



    model.to(device)


    if not args.onlytest:

        train_dataset = BrainHDF5Dataset(path=args.h5,set_name='train', level='0',csv=args.csv,task='fsl')
        test_dataset = BrainHDF5Dataset(path=args.valh5,set_name='test', level='0',csv=args.valcsv,task='fsl')

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=args.batch,
                                    drop_last=False, shuffle=True)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.batch,
                                                drop_last=False, shuffle=False)

        optimizer = optim.SGD(model.parameters(),
                    lr=args.learning_rate,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    nesterov=True)

        criterion = torch.nn.CrossEntropyLoss()


        for epoch in range(0,args.end_epoch):
            adjust_learning_rate(args, optimizer, epoch)
            cnn_train(train_loader, model, criterion, optimizer, epoch,a_device=device)

            # tensorboard logger
            # print('loss: {:.4f}  epoch: {}'.format(loss, epoch))
            print('learning_rate: {:.4f}  epoch: {}'.format(optimizer.param_groups[0]['lr'], epoch))

            if epoch % args.save_freq == 0:
                # run a test
                print('TEST...')
                test_model(test_loader,model,a_device=device,out_probs = args.out_csv)

                save_file = os.path.join(
                    args.save_folder,'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(model, optimizer, args, epoch, save_file)

        # save the last model
        if epoch % args.save_freq != 0:
            save_file = os.path.join(
                args.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, args, args.end_epoch, save_file)

    else:
        print('Testing ONLY....')
        test_dataset = BrainHDF5Dataset(path=args.h5,set_name='test', level='0',csv=args.csv,task='fsl')
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                        batch_size=args.batch,
                                        drop_last=False, shuffle=False)

        test_model(test_loader,model,a_device=device,out_probs = args.out_csv)
        print('Testing is done...')
        







import argparse
parser = argparse.ArgumentParser(description='Configurations')

parser.add_argument('--batch', type=int, help='how large is a batch', default=150)
parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.01)
parser.add_argument('--device', type=int, help='how many devices', default=3)
parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')


parser.add_argument('--csv',
                    help='path for csv partition',
                    default='', type=str)

parser.add_argument('--valcsv',
                    help='path for csv partition',
                    default='', type=str)

parser.add_argument('--out_csv',
                    help='path for csv partition',
                    default='', type=str)

parser.add_argument('--h5',
                    help='path for h5 file',
                    default='', type=str)

parser.add_argument('--valh5',
                    help='path for h5 file',
                    default='', type=str)


parser.add_argument('--lr_decay_epochs', type=str, default='5,10,25,50',
                    help='where to decay lr, can be a list')

parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')

parser.add_argument('--exp_decay_rate', type=float, default=0.95,
                    help='decay rate for learning rate')

parser.add_argument('--lr_scheduling', type=str, choices=['cosine', 'exp_decay', 'adam', 'warmup'],default='warmup',
                    help='what learning rate scheduling to use')
parser.add_argument('--warmup_percent', type=float, default=0.33,
                    help='percent of epochs that used for warmup')

parser.add_argument('--momentum', type=float, default=0.95,
                    help='momentum')
parser.add_argument('--end_epoch', type=int, help='max epoch', default=20)
parser.add_argument('--save_freq', type=int, default=1, help='save frequency')

parser.add_argument('--onlytest', action='store_true')

parser.add_argument('--save_folder',
                    help='string key for saving best model',
                    default='', type=str)


parser.add_argument('--min_lr', type=float, default=1e-5,
                    help='SGD min learning rate')

parser.add_argument('--model', type=str, default='SWIN',
                    help='CNN or SWIN')

parser.add_argument('--test_model_path',
                    help='string key for saving best model',
                    default='', type=str)


args = parser.parse_args()

if __name__ == '__main__':
    # get options

    save_folder = main(args)
    print("ckpt#{}".format(save_folder))
