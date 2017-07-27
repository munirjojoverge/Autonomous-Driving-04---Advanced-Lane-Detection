'''
##__________________________________________________________________________
##                              VIDEO PROCESSING
##
## This file defines main running function to produce the video outputs we are 
## required to present for this assignment.
## We will use the CameraCalibration (calibrate and undistort) and LaneDetector 
## (with all the image utils -sobel, color therholding, line detection and fitting,
## curbature calculation, camera position wrt center lane, and adding all this info
## on the images) to produce a video.
## 
##
## Created on March 20, 2017
## author: MUNIR JOJO-VERGE
##
##__________________________________________________________________________
'''

'''
##__________________________________________________________________________
##  LIBRARIES
##__________________________________________________________________________
'''
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

'''
##__________________________________________________________________________
##  MY LIBRARIES
##__________________________________________________________________________
'''
from CameraCalibration import camera
from LaneFinder import LaneFinder

'''
##__________________________________________________________________________
##  CONSTANTS
##__________________________________________________________________________
'''
FRAME_MEMORY = 7
OFFSET = 250

#VIDEOS = [ "../videos/project_video.mp4",
#VIDEOS = [  "../videos/challenge_video.mp4"]
VIDEOS = [ "../videos/harder_challenge_video.mp4"]

HarderVideo = True

## For the harder Video let's try a 
if HarderVideo:
    SRC = np.float32([
        (50, 700),
        (50, 550),
        (1280-50, 550),
        (1280-50, 700)])
    Y_CUTOFF = 550
else:
    SRC = np.float32([
        (120, 700),
        (510, 465),
        (1280-510, 465),
        (1280-120, 700)])
    Y_CUTOFF = 400

DST = np.float32([
    (SRC[0][0] + OFFSET, 720),
    (SRC[0][0] + OFFSET, 0),
    (SRC[-1][0] - OFFSET, 0),
    (SRC[-1][0] - OFFSET, 720)])

'''
##__________________________________________________________________________
##  MAIN
##__________________________________________________________________________
'''
if __name__ == '__main__':
    camera = camera()    
    
    ld = LaneFinder(SRC, DST, n_frames=FRAME_MEMORY, camera=camera, warp_offset=OFFSET, y_cutoff=Y_CUTOFF)
    
    for video in VIDEOS:
        clip1 = VideoFileClip(video)
        project_clip = clip1.fl_image(ld.process_frame)

        project_output = video[:-4] + '_MunirJojoVerge.mp4'
        project_clip.write_videofile(project_output, audio=False)
