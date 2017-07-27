'''
##__________________________________________________________________________
##                              PERSPECTIVE TRANSFORMER
##
## This file defines the Perspective Class (properties and methods)
## exactly as we did on the Notebook used as a writeup. 
## This notebook should be use as a refernce for this file.
## "Advanced-Lane-Finding-Submission.ipynb"
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
import cv2


class perspective:
    # Define the Properties and the Constructor
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)
    
    # Methods
    def warp(self, img):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR)

    def inv_warp(self, img):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.M_inv, img_size, flags=cv2.INTER_LINEAR)