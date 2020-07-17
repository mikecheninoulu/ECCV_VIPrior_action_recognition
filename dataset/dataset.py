from __future__ import division
from torch.utils.data import Dataset, DataLoader
import getpass
import os
import socket
import numpy as np
from .preprocess_data import *
from PIL import Image, ImageFilter
import pickle
import glob
#import dircache
import pdb


def get_test_video(opt, frame_path, of_frame_path, rp_frame_path, Total_frames):
    """
        Args:
            opt         : config options
            frame_path  : frames of video frames
            Total_frames: Number of frames in the video
        Returns:
            list(frames) : list of all video frames
        """

    clip = []

    if opt.modality == 'RGB':
        sn = opt.sample_duration
        f = lambda n: [(lambda n, arr: n if arr == [] else random.choice(arr))(n * i / sn,
        range(int(n * i / sn), max(int(n * i / sn) + 1, int(n * (i + 1) / sn)))) for i in range(sn)]

        sl = f(Total_frames)
        frams = []
        for a in sl:
            #img = transform(accimage.Image(os.path.join(imgs_path, "%04d.jpg" % a)))
            # img_path = os.path.join(imgs_path, "%05d.jpg" % (a+1))
            # #print(img_path)
            # img = transform(Image.open(img_path))
            # frams.append(self.transform(img).view(3, 112, 112, 1))
            im   = Image.open(os.path.join(frame_path, '%05d.jpg'%(a+1)))
            clip.append(im.copy())
            im.close()

    elif opt.modality == 'Flow':
        sn = opt.sample_duration
        f = lambda n: [(lambda n, arr: n if arr == [] else random.choice(arr))(n * i / sn,
        range(int(n * i / sn), max(int(n * i / sn) + 1, int(n * (i + 1) / sn)))) for i in range(sn)]

        sl = f(Total_frames)
        frams = []
        for a in sl:
            #img = transform(accimage.Image(os.path.join(imgs_path, "%04d.jpg" % a)))
            # img_path = os.path.join(imgs_path, "%05d.jpg" % (a+1))
            # #print(img_path)
            # img = transform(Image.open(img_path))
            # frams.append(self.transform(img).view(3, 112, 112, 1))
            im_x = Image.open(os.path.join(of_frame_path, 'TVL1jpg_x_%06d.jpg'%(a+1)))
            im_y = Image.open(os.path.join(of_frame_path, 'TVL1jpg_y_%06d.jpg'%(a+1)))
            clip.append(im_x.copy())
            clip.append(im_y.copy())
            im_x.close()
            im_y.close()

    elif  opt.modality == 'RGB_Flow':
        sn = opt.sample_duration

        f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn, range(int(n * i / sn),
        max(int(n * i / sn) + 1,int(n * (i + 1) / sn))))for i in range(sn)]
        sl = f(Total_frames)
        frams = []
        for a in sl:
            #img = transform(accimage.Image(os.path.join(imgs_path, "%04d.jpg" % a)))
            # img_path = os.path.join(imgs_path, "%05d.jpg" % (a+1))
            # #print(img_path)
            # img = transform(Image.open(img_path))
            # frams.append(self.transform(img).view(3, 112, 112, 1))
            im   = Image.open(os.path.join(frame_path, '%05d.jpg'%(a+4)))
            im_x = Image.open(os.path.join(of_frame_path, 'TVL1jpg_x_%06d.jpg'%(a+4)))
            im_y = Image.open(os.path.join(of_frame_path, 'TVL1jpg_y_%06d.jpg'%(a+4)))
            clip.append(im.copy())
            clip.append(im_x.copy())
            clip.append(im_y.copy())
            im.close()
            im_x.close()
            im_y.close()

    elif  opt.modality == 'RGB_Flow_rp':
        sn = opt.sample_duration

        f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn, range(int(n * i / sn),
        max(int(n * i / sn) + 1,int(n * (i + 1) / sn))))for i in range(sn)]
        sl = f(Total_frames)
        frams = []
        for a in sl:
            im   = Image.open(os.path.join(frame_path, '%05d.jpg'%(a+4)))
            im_rp  = Image.open(os.path.join(rp_frame_path, '%05d.jpg'%(a+4)))
            im_x = Image.open(os.path.join(of_frame_path, 'TVL1jpg_x_%06d.jpg'%(a+4)))
            im_y = Image.open(os.path.join(of_frame_path, 'TVL1jpg_y_%06d.jpg'%(a+4)))

            clip.append(im.copy())
            clip.append(im_rp.copy())
            clip.append(im_x.copy())
            clip.append(im_y.copy())

            im.close()
            im_x.close()
            im_y.close()
            im_rp.close()
    return clip

def get_train_video(opt, frame_path, of_frame_path, rp_frame_path, Total_frames):
    """
        Chooses a random clip from a video for training/ validation
        Args:
            opt         : config options
            frame_path  : frames of video frames
            Total_frames: Number of frames in the video
        Returns:
            list(frames) : random clip (list of frames of length sample_duration) from a video for training/ validation
        """
    clip = []
    # choosing a random frame
    if Total_frames <= opt.sample_duration:
        loop = 1
        if Total_frames<1:
            print('wrong')
            print(Total_frames)
            print(frame_path)
        start_frame = np.random.randint(0, Total_frames)

    else:
        start_frame = np.random.randint(0, Total_frames - opt.sample_duration)

    if opt.modality == 'RGB':
        sn = opt.sample_duration
        f = lambda n: [(lambda n, arr: n if arr == [] else random.choice(arr))(n * i / sn,
        range(int(n * i / sn), max(int(n * i / sn) + 1, int(n * (i + 1) / sn)))) for i in range(sn)]

        sl = f(Total_frames)
        frams = []
        for a in sl:
            #img = transform(accimage.Image(os.path.join(imgs_path, "%04d.jpg" % a)))
            # img_path = os.path.join(imgs_path, "%05d.jpg" % (a+1))
            # #print(img_path)
            # img = transform(Image.open(img_path))
            # frams.append(self.transform(img).view(3, 112, 112, 1))
            im   = Image.open(os.path.join(frame_path, '%05d.jpg'%(a+1)))
            clip.append(im.copy())
            im.close()

    elif opt.modality == 'Flow':
        sn = opt.sample_duration
        f = lambda n: [(lambda n, arr: n if arr == [] else random.choice(arr))(n * i / sn,
        range(int(n * i / sn), max(int(n * i / sn) + 1, int(n * (i + 1) / sn)))) for i in range(sn)]

        sl = f(Total_frames)
        frams = []
        for a in sl:
            #img = transform(accimage.Image(os.path.join(imgs_path, "%04d.jpg" % a)))
            # img_path = os.path.join(imgs_path, "%05d.jpg" % (a+1))
            # #print(img_path)
            # img = transform(Image.open(img_path))
            # frams.append(self.transform(img).view(3, 112, 112, 1))
            im_x = Image.open(os.path.join(of_frame_path, 'TVL1jpg_x_%06d.jpg'%(a+1)))
            im_y = Image.open(os.path.join(of_frame_path, 'TVL1jpg_y_%06d.jpg'%(a+1)))
            clip.append(im_x.copy())
            clip.append(im_y.copy())
            im_x.close()
            im_y.close()


    elif  opt.modality == 'RGB_Flow':
        sn = opt.sample_duration
        f = lambda n: [(lambda n, arr: n if arr == [] else random.choice(arr))(n * i / sn,
        range(int(n * i / sn), max(int(n * i / sn) + 1, int(n * (i + 1) / sn)))) for i in range(sn)]

        sl = f(Total_frames)
        frams = []
        for a in sl:
            #img = transform(accimage.Image(os.path.join(imgs_path, "%04d.jpg" % a)))
            # img_path = os.path.join(imgs_path, "%05d.jpg" % (a+1))
            # #print(img_path)
            # img = transform(Image.open(img_path))
            # frams.append(self.transform(img).view(3, 112, 112, 1))
            im   = Image.open(os.path.join(frame_path, '%05d.jpg'%(a+4)))
            im_x = Image.open(os.path.join(of_frame_path, 'TVL1jpg_x_%06d.jpg'%(a+4)))
            im_y = Image.open(os.path.join(of_frame_path, 'TVL1jpg_y_%06d.jpg'%(a+4)))
            clip.append(im.copy())
            clip.append(im_x.copy())
            clip.append(im_y.copy())
            im.close()
            im_x.close()
            im_y.close()

    elif  opt.modality == 'RGB_Flow_rp':
        sn = opt.sample_duration
        f = lambda n: [(lambda n, arr: n if arr == [] else random.choice(arr))(n * i / sn,
        range(int(n * i / sn), max(int(n * i / sn) + 1, int(n * (i + 1) / sn)))) for i in range(sn)]

        sl = f(Total_frames)
        frams = []
        for a in sl:
            #img = transform(accimage.Image(os.path.join(imgs_path, "%04d.jpg" % a)))
            # img_path = os.path.join(imgs_path, "%05d.jpg" % (a+1))
            # #print(img_path)
            # img = transform(Image.open(img_path))
            # frams.append(self.transform(img).view(3, 112, 112, 1))
            im   = Image.open(os.path.join(frame_path, '%05d.jpg'%(a+4)))
            im_rp  = Image.open(os.path.join(rp_frame_path, '%05d.jpg'%(a+4)))
            im_x = Image.open(os.path.join(of_frame_path, 'TVL1jpg_x_%06d.jpg'%(a+4)))
            im_y = Image.open(os.path.join(of_frame_path, 'TVL1jpg_y_%06d.jpg'%(a+4)))

            clip.append(im.copy())
            clip.append(im_rp.copy())
            clip.append(im_x.copy())
            clip.append(im_y.copy())

            im.close()
            im_x.close()
            im_y.close()
            im_rp.close()

    return clip


class UCF101_test(Dataset):
    """UCF101 Dataset"""
    def __init__(self, train, opt, split=None):
        """
        Args:
            opt   : config options
            train : 0 for testing, 1 for training, 2 for validation
            split : 1,2,3
        Returns:
            (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.train_val_test = train
        self.opt = opt

        with open(os.path.join(self.opt.annotation_path, "mod-ucf101-classInd.txt")) as lab_file:
            self.lab_names = [line.strip('\n').split(' ')[1] for line in lab_file]

        # print(self.lab_names)

        with open(os.path.join(self.opt.annotation_path, "mod-ucf101-classInd.txt")) as lab_file:
            index = [int(line.strip('\n').split(' ')[0]) for line in lab_file]

        # print(index)

        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 101

        # indexes for training/test set
        self.data = [] # (filename , lab_id)

        if self.train_val_test == 1:
            if self.opt.with_valid == 0:
                labelfile = "mod-ucf101-train.txt"
            elif self.opt.with_valid == 1:
                labelfile = "mod-ucf101-train_all.txt"
            else:
                labelfile = "mod-ucf101-train1.txt"
            f = open(os.path.join(self.opt.annotation_path, labelfile), 'r')
            for line in f:
                # print(line)
                [video_ID, video_cls] = line.strip('\n').split(' ')
                if os.path.exists(os.path.join(self.opt.frame_dir , video_ID[:-4]))== True:
                    self.data.append((video_ID[:-4], int(video_cls)-1))
            f.close()
        elif self.train_val_test == 2:
            labelfile = "mod-ucf101-test_label.txt"
            index = 0
            f = open(os.path.join(self.opt.annotation_path, labelfile), 'r')
            for line in f:
                if index % 5 ==0:
                    # print(line)
                    [video_ID, video_cls] = line.strip('\n').split(' ')
                    if os.path.exists(os.path.join(self.opt.frame_dir , video_ID[:-4]))== True:
                        self.data.append((video_ID[:-4], int(video_cls)-1))
                index +=1
            f.close()
        else:
            labelfile = "mod-ucf101-test_label.txt"
            f = open(os.path.join(self.opt.annotation_path, labelfile), 'r')
            for line in f:
                # print(line)
                [video_ID, video_cls] = line.strip('\n').split(' ')
                if os.path.exists(os.path.join(self.opt.frame_dir , video_ID[:-4]))== True:
                    self.data.append((video_ID[:-4], int(video_cls)-1))
            f.close()

        # print('check')

        # print(os.path.exists(self.opt.annotation_path+labelfile))


        print(len(self.data))

    def __len__(self):
        '''
        returns number of test set
        '''
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label_id = video[1]
        frame_path = os.path.join(self.opt.frame_dir, video[0])
        of_frame_path = os.path.join(self.opt.offrame_dir, video[0])
        rp_frame_path = os.path.join(self.opt.rpframe_dir, video[0])

        if self.opt.modality == 'RGB_Flow':
            # print(of_frame_path)
            if not os.path.exists(of_frame_path):
                print('holy shit')

            Total_frames1 = len(glob.glob(glob.escape(frame_path) +  '/0*.jpg'))
            # print(Total_frames)
            Total_frames2 = len(glob.glob(glob.escape(of_frame_path) +  '/TVL1jpg_y_*.jpg'))

            Total_frames = min(Total_frames1,Total_frames2)
            # print(Total_frames)
        elif self.opt.modality == 'RGB':

            Total_frames = len(glob.glob(glob.escape(frame_path) +  '/0*.jpg'))
            # print(Total_frames)
        elif self.opt.modality == 'Flow':

            Total_frames = len(glob.glob(glob.escape(of_frame_path) +  '/TVL1jpg_y_*.jpg'))
            # print(Total_frames)
        elif self.opt.modality == 'RGB_Flow_rp':
            # print(of_frame_path)

            Total_frames = len(glob.glob(glob.escape(rp_frame_path) +  '/0*.jpg'))

        if self.train_val_test == 0:
            clip = get_test_video(self.opt, frame_path, of_frame_path, rp_frame_path, Total_frames)
            video_ID = video[0].split('/')[-1]
            return((scale_crop(clip, self.train_val_test, self.opt), video_ID+'.avi', label_id))
        else:
            clip = get_train_video(self.opt, frame_path, of_frame_path, rp_frame_path, Total_frames)
            return((scale_crop(clip, self.train_val_test, self.opt), label_id))
