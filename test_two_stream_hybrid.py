from dataset.dataset import *
from torch.utils.data import Dataset, DataLoader
import getpass
import os
import socket
import numpy as np
from dataset.preprocess_data import *
import torch
from models.model import generate_model
from opts import parse_opts
from torch.autograd import Variable
import torch.nn.functional as F
import time
import sys
from utils import AverageMeter, calculate_accuracy, calculate_accuracy_video
import random
import pdb
from  tqdm import tqdm


def test():

    opt = parse_opts()

    opt.modality = 'RGB_Flow'
    opt.dataset = 'UCF101'
    opt.model = 'tc3d'
    opt.model_depth = 101
    opt.log = 1
    opt.only_RGB = False
    opt.n_classes = 101
    opt.batch_size = 32
    opt.sample_duration = 12
    #opt.resume_path2 = "/home/haoyu/Documents/6_ECCV_competition/vipriors-challenges-toolkit-master/action-recognition/baselines/MARS-master/results/server_results/UCF101/with validation/Flow_train_batch64_sample112_clip12_modeltc3d101_epoch152_acc0.7428977272727273.pth"
    opt.resume_path1 = "/home/haoyu/Documents/6_ECCV_competition/vipriors-challenges-toolkit-master/action-recognition/baselines/MARS-threestream_hybrid/results/RGB_Flow_train_batch64_sample112_clip12_modeltc3d50_epoch102_acc0.6988636363636364_withvalid_1.pth"
    opt.resume_path2 = "/home/haoyu/Documents/6_ECCV_competition/vipriors-challenges-toolkit-master/action-recognition/baselines/MARS-threestream_hybrid/results/Flow_train_batch64_sample112_clip12_modeltc3d101_epoch152_acc0.7428977272727273.pth"
    #opt.resume_path1 = "/home/haoyu/Documents/6_ECCV_competition/vipriors-challenges-toolkit-master/action-recognition/baselines/MARS-master/results/server_results/UCF101/with validation/RGB_train_batch64_sample112_clip12_modeltc3d101_epoch118_acc0.5553977272727273.pth"
    opt.frame_dir = '/home/haoyu/Documents/6_ECCV_competition/vipriors-challenges-toolkit-master/action-recognition/data/mod-ucf101/rp/'
    opt.offrame_dir = '/home/haoyu/Documents/6_ECCV_competition/vipriors-challenges-toolkit-master/action-recognition/data/mod-ucf101/opticalflows/'
    opt.annotation_path = "/home/haoyu/Documents/6_ECCV_competition/vipriors-challenges-toolkit-master/action-recognition/data/mod-ucf101/annotations/"
    opt.result_path =  "results/"
    opt.rpframe_dir = ''

    print(opt)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

    print("Preprocessing validation data ...")
    data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 0, opt = opt)
    print("Length of validation data = ", len(data))

    print("Preparing datatloaders ...")
    val_dataloader = DataLoader(data, batch_size = 1, shuffle=False, num_workers = opt.n_workers, pin_memory = True, drop_last=False)
    print("Length of validation datatloader = ",len(val_dataloader))

    result_path = "{}/{}/".format(opt.result_path, opt.dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # define the model
    print("Loading models... ", opt.model, opt.model_depth)
    opt.model_depth = 50
    opt.input_channels = 5
    model1, parameters1 = generate_model(opt)

    # if testing RGB+Flow streams change input channels
    if not opt.only_RGB:
        opt.input_channels = 2

    opt.model_depth = 101
    opt.input_channels = 2
    model2, parameters2 = generate_model(opt)

    if opt.resume_path1:
        print('loading checkpoint {}'.format(opt.resume_path1))
        checkpoint = torch.load(opt.resume_path1)
        # assert opt.arch == checkpoint['arch']
        model1.load_state_dict(checkpoint['state_dict'])
    if opt.resume_path2:
        print('loading checkpoint {}'.format(opt.resume_path2))
        checkpoint = torch.load(opt.resume_path2)
        # assert opt.arch == checkpoint['arch']
        model2.load_state_dict(checkpoint['state_dict'])

    model1.eval()
    model2.eval()

    accuracies1 = AverageMeter()
    accuracies2 = AverageMeter()
    accuracies3 = AverageMeter()
    accuracies4 = AverageMeter()
    accuracies5 = AverageMeter()
    accuracies6 = AverageMeter()
    accuracies7 = AverageMeter()
    accuracies8 = AverageMeter()
    accuracies9 = AverageMeter()

    if opt.log:
        f = open(os.path.join(result_path, "test_{}{}_{}_{}_{}_{}.txt".format( opt.model, opt.model_depth, opt.dataset, opt.split, opt.modality, opt.sample_duration)), 'w+')
        f.write(str(opt))
        f.write('\n')
        f.flush()

    submission = open(os.path.join(result_path, "submit_hybrid_{}{}_{}_{}_{}_{}.txt".format( opt.model, opt.model_depth, opt.dataset, opt.split, opt.modality, opt.sample_duration)), 'w+')

    RGB_candidate_top5 = []
    Flow_candidate_top5 = []

    submission1 = []
    submission2 = []
    submission3 = []
    submission4 = []
    submission5 = []
    submission6 = []
    submission6 = []
    submission7 = []
    submission8 = []
    submission9 = []

    with torch.no_grad():
        for i, (clip,videoID, label) in tqdm(enumerate(val_dataloader)):
            clip = torch.squeeze(clip)
            if opt.only_RGB:
                inputs = torch.Tensor(int(clip.shape[1]/opt.sample_duration), 3, opt.sample_duration, opt.sample_size, opt.sample_size)
                for k in range(inputs.shape[0]):
                    inputs[k,:,:,:,:] = clip[:,k*opt.sample_duration:(k+1)*opt.sample_duration,:,:]

                inputs_var1 = Variable(inputs)
                inputs_var2 = Variable(inputs)
            else:
                RGB_clip  = clip
                Flow_clip = clip[3:,:,:,:]
                inputs1 = torch.Tensor(int(RGB_clip.shape[1]/opt.sample_duration), 5, opt.sample_duration, opt.sample_size, opt.sample_size)
                inputs2 = torch.Tensor(int(Flow_clip.shape[1]/opt.sample_duration), 2, opt.sample_duration, opt.sample_size, opt.sample_size)
                for k in range(inputs1.shape[0]):
                    inputs1[k,:,:,:,:] = RGB_clip[:,k*opt.sample_duration:(k+1)*opt.sample_duration,:,:]
                    inputs2[k,:,:,:,:] = Flow_clip[:,k*opt.sample_duration:(k+1)*opt.sample_duration,:,:]
                inputs_var1 = Variable(inputs1)
                inputs_var2 = Variable(inputs2)


            outputs_var1= model1(inputs_var1)
            outputs_var2= model2(inputs_var2)
            # print(outputs_var1.shape)
            #outputs_var_0_4 = outputs_var1 + outputs_var2*3/2


            outputs_var1= model1(inputs_var1)
            outputs_var2= model2(inputs_var2)
            # print(outputs_var1.shape)
            outputs_var_0_1 = outputs_var1 + outputs_var2*7/3
            outputs_var_0_2 = outputs_var1 + outputs_var2*6.75/3.75
            outputs_var_0_3 = outputs_var1 + outputs_var2*6.5/3.5
            outputs_var_0_4 = outputs_var1 + outputs_var2*6.25/3.75
            outputs_var_0_5 = outputs_var1 + outputs_var2*6/4
            outputs_var_0_6 = outputs_var1 + outputs_var2*5.75/4.25
            outputs_var_0_7 = outputs_var1 + outputs_var2*5.5/4.5
            outputs_var_0_8 = outputs_var1 + outputs_var2*5.25/4.75
            outputs_var_0_9 = outputs_var1 + outputs_var2*5/5

            pred5_1 = np.array(torch.mean(outputs_var_0_1, dim=0, keepdim=True).topk(5, 1, True)[1].cpu().data[0])
            pred5_2 = np.array(torch.mean(outputs_var_0_2, dim=0, keepdim=True).topk(5, 1, True)[1].cpu().data[0])
            pred5_3 = np.array(torch.mean(outputs_var_0_3, dim=0, keepdim=True).topk(5, 1, True)[1].cpu().data[0])
            pred5_4 = np.array(torch.mean(outputs_var_0_4, dim=0, keepdim=True).topk(5, 1, True)[1].cpu().data[0])
            pred5_5 = np.array(torch.mean(outputs_var_0_5, dim=0, keepdim=True).topk(5, 1, True)[1].cpu().data[0])
            pred5_6 = np.array(torch.mean(outputs_var_0_6, dim=0, keepdim=True).topk(5, 1, True)[1].cpu().data[0])
            pred5_7 = np.array(torch.mean(outputs_var_0_7, dim=0, keepdim=True).topk(5, 1, True)[1].cpu().data[0])
            pred5_8 = np.array(torch.mean(outputs_var_0_8, dim=0, keepdim=True).topk(5, 1, True)[1].cpu().data[0])
            pred5_9 = np.array(torch.mean(outputs_var_0_9, dim=0, keepdim=True).topk(5, 1, True)[1].cpu().data[0])
            # print(outputs_var)
            acc1 = float(pred5_1[0] == label[0])
            acc2 = float(pred5_2[0] == label[0])
            acc3 = float(pred5_3[0] == label[0])
            acc4 = float(pred5_4[0] == label[0])
            acc5 = float(pred5_5[0] == label[0])
            acc6 = float(pred5_6[0] == label[0])
            acc7 = float(pred5_7[0] == label[0])
            acc8 = float(pred5_8[0] == label[0])
            acc9 = float(pred5_9[0] == label[0])

            accuracies1.update(acc1, 1)
            accuracies2.update(acc2, 1)
            accuracies3.update(acc3, 1)
            accuracies4.update(acc4, 1)
            accuracies5.update(acc5, 1)
            accuracies6.update(acc6, 1)
            accuracies7.update(acc7, 1)
            accuracies8.update(acc8, 1)
            accuracies9.update(acc9, 1)
            # line = "Video[" + str(i) + "\t video = " + str(accuracies5.avg)
            # print(line)
            # line = "Video[" + str(i) + "] : \t top5 " + str(pred5) + "\t top1 = " + str(pred5[0]) +  "\t true = " +str(int(label[0])) + "\t video = " + str(accuracies.avg)
            # print(line)
            resulttop5_1  = videoID[0] +' ' + str(pred5_1[0]+1)+' ' + str(pred5_1[1]+1)+' ' + str(pred5_1[2]+1)+' ' + str(pred5_1[3]+1)+' ' + str(pred5_1[4]+1)
            resulttop5_2  = videoID[0] +' ' + str(pred5_2[0]+1)+' ' + str(pred5_2[1]+1)+' ' + str(pred5_2[2]+1)+' ' + str(pred5_2[3]+1)+' ' + str(pred5_2[4]+1)
            resulttop5_3  = videoID[0] +' ' + str(pred5_3[0]+1)+' ' + str(pred5_3[1]+1)+' ' + str(pred5_3[2]+1)+' ' + str(pred5_3[3]+1)+' ' + str(pred5_3[4]+1)
            resulttop5_4  = videoID[0] +' ' + str(pred5_4[0]+1)+' ' + str(pred5_4[1]+1)+' ' + str(pred5_4[2]+1)+' ' + str(pred5_4[3]+1)+' ' + str(pred5_4[4]+1)
            resulttop5_5  = videoID[0] +' ' + str(pred5_5[0]+1)+' ' + str(pred5_5[1]+1)+' ' + str(pred5_5[2]+1)+' ' + str(pred5_5[3]+1)+' ' + str(pred5_5[4]+1)
            resulttop5_6  = videoID[0] +' ' + str(pred5_6[0]+1)+' ' + str(pred5_6[1]+1)+' ' + str(pred5_6[2]+1)+' ' + str(pred5_6[3]+1)+' ' + str(pred5_6[4]+1)
            resulttop5_7  = videoID[0] +' ' + str(pred5_7[0]+1)+' ' + str(pred5_7[1]+1)+' ' + str(pred5_7[2]+1)+' ' + str(pred5_7[3]+1)+' ' + str(pred5_7[4]+1)
            resulttop5_8  = videoID[0] +' ' + str(pred5_8[0]+1)+' ' + str(pred5_8[1]+1)+' ' + str(pred5_8[2]+1)+' ' + str(pred5_8[3]+1)+' ' + str(pred5_8[4]+1)
            resulttop5_9  = videoID[0] +' ' + str(pred5_9[0]+1)+' ' + str(pred5_9[1]+1)+' ' + str(pred5_9[2]+1)+' ' + str(pred5_9[3]+1)+' ' + str(pred5_9[4]+1)


            submission1.append(resulttop5_1)
            submission2.append(resulttop5_2)
            submission3.append(resulttop5_3)
            submission4.append(resulttop5_4)
            submission5.append(resulttop5_5)
            submission6.append(resulttop5_6)
            submission7.append(resulttop5_7)
            submission8.append(resulttop5_8)
            submission9.append(resulttop5_9)


            # print(resulttop5)


    print("Video accuracy = ", accuracies1.avg)
    print("Video accuracy = ", accuracies2.avg)
    print("Video accuracy = ", accuracies3.avg)
    print("Video accuracy = ", accuracies4.avg)
    print("Video accuracy = ", accuracies5.avg)
    print("Video accuracy = ", accuracies6.avg)
    print("Video accuracy = ", accuracies7.avg)
    print("Video accuracy = ", accuracies8.avg)
    print("Video accuracy = ", accuracies9.avg)

    acc_list = []
    acc_list.append(accuracies1.avg)
    acc_list.append(accuracies2.avg)
    acc_list.append(accuracies3.avg)
    acc_list.append(accuracies4.avg)
    acc_list.append(accuracies5.avg)
    acc_list.append(accuracies6.avg)
    acc_list.append(accuracies7.avg)
    acc_list.append(accuracies8.avg)
    acc_list.append(accuracies9.avg)

    best_index = acc_list.index(max(acc_list))
    if best_index == 0:
        submission = submission1
    elif best_index == 1:
        submission = submission2
    elif best_index == 2:
        submission = submission3
    elif best_index == 3:
        submission = submission4
    elif best_index == 4:
        submission = submission5
    elif best_index == 5:
        submission = submission6
    elif best_index == 6:
        submission = submission7
    elif best_index == 7:
        submission = submission8
    elif best_index == 8:
        submission = submission9
    else:
        print("Invalid month")
            # submission.write(resulttop5 + '\n')
            # submission.flush()
            # if opt.log:
            #
            #     f.write(line + '\n')
            #     f.flush()

    # print("Video accuracy = ", accuracies.avg)
    # line = "Video accuracy = " + str(accuracies.avg) + '\n'
    # if opt.log:
    #     f.write(line)

    submission_file = open(os.path.join(result_path, "best_submit_{}{}_{}_{}_{}_{}.txt".format( opt.model, opt.model_depth, opt.dataset, opt.split, opt.modality, opt.sample_duration)), 'w+')
    for lines in submission:
        submission_file.write(lines + '\n')
    submission_file.flush()

if __name__=="__main__":
    test()
