'''
For HMDB51 and UCF101 datasets:

Code extracts frames from video at a rate of 25fps and scaling the
larger dimension of the frame is scaled to 256 pixels.
After extraction of all frames write a "done" file to signify proper completion
of frame extraction.

Usage:
  python extract_frames.py video_dir frame_dir

  video_dir => path of video files
  frame_dir => path of extracted jpg frames

'''
import shutil
import sys, os, pdb
import numpy as np
import subprocess
from tqdm import tqdm


def extract(vid_dir, frame_dir, start, end, redo=False):
    video_list = []
    video_list = os.listdir(vid_dir)

    #print("Classes =", class_list)

    shutil.rmtree(frame_dir)
    os.mkdir(frame_dir)

    for videoID in tqdm(video_list):
        # print(videoID)
        # print(int(label))
        outdir = os.path.join(frame_dir, videoID[:-4])
        # print(outdir)

        # Checking if frames already extracted
        if os.path.isfile(os.path.join(outdir, 'done') ) and not redo: continue
        try:
            os.system('mkdir -p "%s"'%(outdir))
            # check if horizontal or vertical scaling factor
            o = subprocess.check_output('ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 "%s"'%(os.path.join(vid_dir, videoID)), shell=True).decode('utf-8')
            lines = o.splitlines()
            width = int(lines[0].split('=')[1])
            height = int(lines[1].split('=')[1])
            resize_str = '-1:256' if width>height else '256:-1'

            # extract frames
            os.system('ffmpeg -i "%s" -r 25 -q:v 2 -vf "scale=%s" "%s"  > /dev/null 2>&1'%( os.path.join(vid_dir, videoID), resize_str, os.path.join(outdir, '%05d.jpg')))
            nframes = len([ fname for fname in os.listdir(outdir) if fname.endswith('.jpg') and len(fname)==9])
            if nframes==0: raise Exception

            os.system('touch "%s"'%(os.path.join(outdir, 'done') ))
        except:
            print("ERROR", label, videoID)

if __name__ == '__main__':
  vid_dir = '/home/haoyu/Documents/6_ECCV_competition/vipriors-challenges-toolkit-master/action-recognition/data/mod-ucf101/videos/'
  frame_dir = '/home/haoyu/Documents/6_ECCV_competition/vipriors-challenges-toolkit-master/action-recognition/data/mod-ucf101/frames/'
  start     = 0
  end       = 2
  extract(vid_dir, frame_dir, start, end, redo=True)
