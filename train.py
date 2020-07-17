from dataset.dataset import *
from torch.utils.data import Dataset, DataLoader
import getpass
import os
import socket
import numpy as np
from dataset.preprocess_data import *
from PIL import Image, ImageFilter
import argparse
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from models.model import generate_model
from opts import parse_opts
from torch.autograd import Variable
import time
import sys
from utils import *
#from utils import AverageMeter, calculate_accuracy
import pdb

#  python train.py --dataset Kinetics --modality RGB --only_RGB \
# --n_classes 400 \
# --batch_size 32 --log 1 --sample_duration 16 \
# --model resnext --model_depth 101  \
# --frame_dir "dataset/Kinetics" \
# --annotation_path "dataset/Kinetics_labels" \
# --t_path "results/"

if __name__=="__main__":
    opt = parse_opts()
    opt.dataset = 'UCF101'
    opt.log = 1
    opt.MARS_alpha = 50
    opt.n_classes = 101
    opt.n_epochs = 200
    if opt.PC == 'local':
        opt.frame_dir = '/home/haoyu/Documents/6_ECCV_competition/vipriors-challenges-toolkit-master/action-recognition/data/mod-ucf101/frames/'
        opt.rp_dir = '/home/haoyu/Documents/6_ECCV_competition/vipriors-challenges-toolkit-master/action-recognition/data/mod-ucf101/rp/'
        opt.offrame_dir = '/home/haoyu/Documents/6_ECCV_competition/vipriors-challenges-toolkit-master/action-recognition/data/mod-ucf101/opticalflows/'
        opt.annotation_path = "/home/haoyu/Documents/6_ECCV_competition/vipriors-challenges-toolkit-master/action-recognition/data/mod-ucf101/annotations/"
        opt.result_path =  "results/"
    elif opt.PC == 'csc':
        opt.rp_dir = '/scratch/project_2003209/rp/'
        opt.frame_dir = '/scratch/project_2003209/frames/'
        opt.offrame_dir = '/scratch/project_2003217/opticalflows/'
        opt.annotation_path = '/scratch/project_2003209/annotations/'
        opt.result_path =  "/scratch/project_2003209/results/"
    elif opt.PC == 'server':
        opt.rp_dir = '/home/micro-gesture/Documents/Haoyu/ECCV/NAS6layer_CenterD_local-20200703T095129Z-001/NAS6layer_CenterD_local/data/rp/'
        opt.frame_dir = '/home/micro-gesture/Documents/Haoyu/ECCV/NAS6layer_CenterD_local-20200703T095129Z-001/NAS6layer_CenterD_local/data/frames/'
        opt.offrame_dir = '/home/micro-gesture/Documents/Haoyu/ECCV/NAS6layer_CenterD_local-20200703T095129Z-001/NAS6layer_CenterD_local/data/opticalflows/'
        opt.annotation_path = "/media/micro-gesture/data/dataset/UCF101mod/annotations/"
        opt.result_path =  "results/"
    else:
        print('wrong machine!')
    # print(opt)

    # print(opt)
    min_acc = 0.3

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    torch.manual_seed(opt.manual_seed)

    print("Preprocessing train data ...")
    train_data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 1, opt = opt)
    print("Length of train data = ", len(train_data))

    print("Preprocessing validation data ...")
    val_data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 2, opt = opt)
    print("Length of validation data = ", len(val_data))

    if opt.modality=='RGB': opt.input_channels = 3
    elif opt.modality=='Flow': opt.input_channels = 2

    print("Preparing datatloaders ...")
    train_dataloader = DataLoader(train_data, batch_size = opt.batch_size, shuffle=True, num_workers = opt.n_workers, pin_memory = True, drop_last=True)
    val_dataloader   = DataLoader(val_data, batch_size = opt.batch_size, shuffle=True, num_workers = opt.n_workers, pin_memory = True, drop_last=True)
    print("Length of train datatloader = ",len(train_dataloader))
    print("Length of validation datatloader = ",len(val_dataloader))

    # define the model
    print("Loading model... ", opt.model, opt.model_depth)
    model, parameters = generate_model(opt)

    criterion = nn.CrossEntropyLoss().cuda()

    if opt.resume_path1:
        print('loading checkpoint {}'.format(opt.resume_path1))
        checkpoint = torch.load(opt.resume_path1)

        assert opt.arch == checkpoint['arch']
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    log_path = os.path.join(opt.result_path, opt.dataset)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if opt.log == 1:
        if opt.pretrain_path:
            epoch_logger = Logger(os.path.join(log_path, 'PreKin_{}_{}_{}_train_batch{}_sample{}_clip{}_nest{}_damp{}_weight_decay{}_manualseed{}_model{}{}_ftbeginidx{}_varLR.log'
                            .format(opt.dataset, opt.split, opt.modality, opt.batch_size, opt.sample_size, opt.sample_duration, opt.nesterov, opt.dampening, opt.weight_decay, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index))
                            ,['epoch', 'loss', 'acc', 'lr'], opt.resume_path1, opt.begin_epoch-1)
            val_logger   = Logger(os.path.join(log_path, 'PreKin_{}_{}_{}_val_batch{}_sample{}_clip{}_nest{}_damp{}_weight_decay{}_manualseed{}_model{}{}_ftbeginidx{}_varLR.log'
                            .format(opt.dataset, opt.split, opt.modality, opt.batch_size, opt.sample_size, opt.sample_duration, opt.nesterov, opt.dampening, opt.weight_decay, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index))
                            ,['epoch', 'loss', 'acc'], opt.resume_path1, opt.begin_epoch-1)
        else:

            epoch_logger = Logger(os.path.join(log_path, '{}_train_batch{}_sample{}_clip{}_model{}_{}_valid_{}.log'
                            .format(opt.modality, opt.batch_size, opt.sample_size, opt.sample_duration, opt.model, opt.model_depth, opt.with_valid))
                            ,['epoch', 'loss', 'acc', 'lr'], opt.resume_path1, opt.begin_epoch-1)
            val_logger   = Logger(os.path.join(log_path, '{}_val_batch{}_sample{}_clip{}_model{}_{}_valid_{}.log'
                            .format(opt.modality, opt.batch_size, opt.sample_size, opt.sample_duration, opt.model, opt.model_depth, opt.with_valid))
                            ,['epoch', 'loss', 'acc'], opt.resume_path1, opt.begin_epoch-1)


    print("Initializing the optimizer ...")
    if opt.pretrain_path:
        opt.weight_decay = 1e-5
        opt.learning_rate = 0.001

    if opt.nesterov: dampening = 0
    else: dampening = opt.dampening

    print("lr = {} \t momentum = {} \t dampening = {} \t weight_decay = {}, \t nesterov = {}"
                .format(opt.learning_rate, opt.momentum, dampening, opt. weight_decay, opt.nesterov))
    print("LR patience = ", opt.lr_patience)


    optimizer = optim.SGD(
        parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)

    if opt.resume_path1 != '':
        optimizer.load_state_dict(torch.load(opt.resume_path1)['optimizer'])

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)

    print('run')
    for epoch in range(opt.begin_epoch, opt.n_epochs + 1):

        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        #pdb.set_trace()
        for i, (inputs, targets) in enumerate(train_dataloader):
            data_time.update(time.time() - end_time)

            targets = targets.cuda(non_blocking=True)
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(train_dataloader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies))

        if opt.log == 1:
            epoch_logger.log({
                'epoch': epoch,
                'loss': losses.avg,
                'acc': accuracies.avg,
                'lr': optimizer.param_groups[0]['lr']
            })

        model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_dataloader):

                # pdb.set_trace()
                data_time.update(time.time() - end_time)
                targets = targets.cuda(non_blocking=True)
                inputs = Variable(inputs)
                targets = Variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                acc = calculate_accuracy(outputs, targets)

                losses.update(loss.item(), inputs.size(0))
                accuracies.update(acc, inputs.size(0))

                batch_time.update(time.time() - end_time)
                end_time = time.time()

                print('Val_Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch,
                        i + 1,
                        len(val_dataloader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        acc=accuracies))

        if opt.log == 1:
            val_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
        scheduler.step(losses.avg)

        if epoch % opt.checkpoint == 0:
            if opt.pretrain_path:
                save_file_path = os.path.join(log_path, 'PreKin_{}_{}_{}_train_batch{}_sample{}_clip{}_nest{}_damp{}_weight_decay{}_manualseed{}_model{}{}_ftbeginidx{}_varLR{}.pth'
                            .format(opt.dataset, opt.split, opt.modality, opt.batch_size, opt.sample_size, opt.sample_duration, opt.nesterov, opt.dampening, opt.weight_decay, opt.manual_seed, opt.model,
                                    opt.model_depth, epoch))
            else:
                if accuracies.avg > min_acc:

                    save_file_path = os.path.join(log_path, '{}_train_batch{}_sample{}_clip{}_model{}{}_epoch{}_acc{}_withvalid_{}.pth'
                                .format(opt.modality, opt.batch_size, opt.sample_size, opt.sample_duration, opt.model, opt.model_depth, epoch, accuracies.avg, opt.with_valid))
                    states = {
                        'epoch': epoch + 1,
                        'arch': opt.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(states, save_file_path)
                    min_acc = accuracies.avg
