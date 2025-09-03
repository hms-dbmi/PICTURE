"""
Modified based on SCAN
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import numpy as np
from utils.utils import AverageMeter, ProgressMeter
import timm
from sklearn import metrics
import pandas as pd


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    print(lr)
    if args.lr_scheduling == 'adam':
        return None
    elif args.lr_scheduling == 'cosine':
        eta_min = lr * (args.lr_decay_rate ** 3)

        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.end_epoch)) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.lr_scheduling == 'exp_decay':
        if epoch == 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(lr, args.min_lr)
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * args.exp_decay_rate, args.min_lr)

    elif args.lr_scheduling == 'warmup':
        assert args.learning_rate >= args.min_lr, "learning rate should >= min lr"
        warmup_epochs = int(args.end_epoch * args.warmup_percent)
        up_slope = (args.learning_rate - args.min_lr) / warmup_epochs
        down_slope = (args.learning_rate - args.min_lr) / (args.end_epoch - warmup_epochs)
        if epoch <= warmup_epochs:
            lr = args.min_lr + up_slope * epoch
        else:
            # lr = args.learning_rate - slope * (epoch - warmup_epochs)
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)

            lr = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * (epoch - warmup_epochs) / (args.end_epoch - warmup_epochs))) / 2

        for param_group in optimizer.param_groups:
            param_group['lr'] = max(lr, args.min_lr)
            print(param_group['lr'])


    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def test_model (testloader, net, no_of_classes = 2,a_device = False, out_probs = None,multi_cut=False):

    net.eval()
    patient_all = []
    predictions_all = []
    label_all = []
    running_corrects = 0

    total_dict = {i: 0 for i in range(no_of_classes)}
    hit_dict = {i: 0 for i in range(no_of_classes)}

    for i, batch in enumerate(testloader):

        images = batch['image'].to(a_device)
        labels = batch['target'].type(torch.LongTensor).to(a_device)
        patients = batch['meta']['patient_name']
        logps = net(images)
        total_dict = {i: total_dict[i] + labels.tolist().count(i) for i in range(no_of_classes)}

        probs = torch.nn.functional.softmax(logps, dim=1)

        # Running count of correctly identified classes
        ps = torch.exp(logps)
        _, predictions = ps.topk(1, dim=1)  # top predictions

        equals = predictions == labels.view(*predictions.shape)

        if len(predictions_all) == 0:
            predictions_all = ps.detach().cpu().numpy()
            label_all = labels.tolist()
            patient_all = list(patients)
            probs_all = probs.detach().cpu().numpy()
        else:
            try:
                predictions_all = np.vstack((predictions_all, ps.detach().cpu().numpy()))
                probs_all = np.vstack((probs_all, probs.detach().cpu().numpy()))
                label_all.extend(labels.tolist())
                patient_all.extend(list(patients))
            except:
                print(ps.detach().cpu().numpy().shape)

        all_hits = equals.view(equals.shape[0]).tolist()  # get all T/F indices
        all_corrects = labels[all_hits]

        hit_dict = {i: hit_dict[i] + all_corrects.tolist().count(i) for i in range(no_of_classes)}
        running_corrects += torch.sum(equals.type(torch.FloatTensor)).item()

    phase_acc = running_corrects / sum(total_dict.values())
    print('All tested patients/slides...',set(patient_all),len(set(patient_all)))
    y_true = label_all.copy()
    y_pred = np.argmax(predictions_all, axis=1)
    print("accuray:", len(np.arange(len(y_true))[y_true == y_pred]) / len(y_true))
    metrics.confusion_matrix(y_true, y_pred)
    print(metrics.classification_report(y_true, y_pred, digits=3))

    # Compute ROC curve and ROC area for each class
    roc_auc = dict()

    fpr, tpr, _ = metrics.roc_curve(y_true, probs_all[:, 1])
    roc_auc[0] = metrics.auc(fpr, tpr)
    print("Binary AUC: {:.4f}".format(roc_auc[0]))
    roc_auc["micro"] = roc_auc[0]
    print("Micro Tile AUC: {:.4f}".format(roc_auc["micro"]))


    pa_auc = patient_aggregation(patient_all, probs_all, label_all, binary=True, SAVE_PATH = out_probs,multi_cut=multi_cut)

    print("\n".join("class: {}\t counts: {}".format(k, v) for k, v in total_dict.items()))
    print("\n".join("class: {}\t counts: {}".format(k, v) for k, v in hit_dict.items()))
    print('Acc: {:.4f} PA-AUC {:.4f}'.format(phase_acc, pa_auc))


def cnn_train(train_loader, model, criterion, optimizer, epoch,a_device):

    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        images = batch['image'].to(a_device)
        targets = batch['target'].type(torch.LongTensor).to(a_device)

        output = model(images)
        loss = criterion(output,targets)
        losses.update(loss.item())
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)





def simclr_train(train_loader, model, criterion, optimizer, epoch):
    """ 
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w) 
        input_ = input_.cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1)
        loss = criterion(output)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def scan_train(train_loader, model, criterion, optimizer, epoch, update_cluster_head_only=False):
    """ 
    Train w/ SCAN-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [total_losses, consistency_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch))

    if update_cluster_head_only:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN

    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = batch['anchor'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)
       
        if update_cluster_head_only: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')

        else: # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            neighbors_output = model(neighbors)     

        # Loss for every head
        total_loss, consistency_loss, entropy_loss = [], [], []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            total_loss_, consistency_loss_, entropy_loss_ = criterion(anchors_output_subhead,
                                                                         neighbors_output_subhead)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            entropy_loss.append(entropy_loss_)

        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(np.mean([v.item() for v in total_loss]))
        consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
        entropy_losses.update(np.mean([v.item() for v in entropy_loss]))

        total_loss = torch.sum(torch.stack(total_loss, dim=0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)