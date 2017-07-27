'''
##__________________________________________________________________________
##                              CAMERA CALIBRATION
##
## This file defines the CAMERA CALIBRATION
##
## The Notebook used as a writeup explains all the steps and the reasons for
## value choices, algorithm choices, etc..
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
import pickle
from glob import glob

import cv2
import numpy as np
from scipy.misc import imresize, imread
from tqdm import tqdm

'''
##__________________________________________________________________________
##  CONTANTS & CONFIGURATION ELEMENTS
##__________________________________________________________________________
'''
# Define the number of ROWS and COLS on the chessboard images we will use to calibrate the camera.
ROWS = 7 # The number of ROWS on the chessboard image. The inside corners in Y direction would be ROWS-1.
COLS = 10 # The number of COLUMS on the chessboard image. The inside corners in X direction would be COLS-1.

CAL_IMAGE_SIZE = (720, 1280, 3)
CAL_IMAGE_PATH = '../camera_cal/calibration*.jpg'
CALIBRATION_PATH = '../camera_cal/camera_calibration.p'

'''
##__________________________________________________________________________
##  camera CLASS
##__________________________________________________________________________
'''
class camera:
    
    def calibrate(self):        
        calibration = get_camera_calibration()
        
        self.objpoints = calibration['objpoints']
        self.imgpoints = calibration['imgpoints']
        self.mtx = calibration['mtx']
        self.dist = calibration['dist']
        
    # Contructor
    def __init__(self):
        """
        Camera Class Constructor
        """
        self.calibrate()
                   
    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)


def calibrate_camera(calib_imgs_path, rows, cols):
    """
    Calculates the camera:
        1) Camera Matrix used for perfective
        2) distortion coefficients
        3) rotation vectors
        4) Translation vectors 
    based on chessboard images
    input parameters:
        calib_imgs_path: Path to the calibration images
        rows: number of inner rows on chessboard
        cols: number of inner columns on chessboard
    returns:
        calibration (look bellow)
    """
    # We are interested in the number of inner corners:
    cols = cols-1
    rows = rows-1
    
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = glob(calib_imgs_path)
    cal_images = np.zeros((len(images), *CAL_IMAGE_SIZE), dtype=np.uint8)

    successfull_cnt = 0
    print("Calibrating camera...")
    for idx, fname in enumerate(tqdm(images, desc='Processing image')):
        img = imread(fname)
        if img.shape[0] != CAL_IMAGE_SIZE[0] or img.shape[1] != CAL_IMAGE_SIZE[1]:
            img = imresize(img, CAL_IMAGE_SIZE)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if ret:
            successfull_cnt += 1

            objpoints.append(objp)
            imgpoints.append(corners)

            img = cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
            cal_images[idx] = img

    print("%s/%s camera calibration images processed." % (successfull_cnt, len(images)))

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, CAL_IMAGE_SIZE[:-1], None, None)

    calibration = {'objpoints': objpoints,
                   'imgpoints': imgpoints,
                   'cal_images': cal_images,
                   'ret': ret,
                   'mtx': mtx,
                   'dist': dist,
                   'rvecs': rvecs,
                   'tvecs': tvecs}

    return calibration


def get_camera_calibration():
    """
    Retrieves the calibration file (CALIBRATION_PATH), but if
    the camera calibration file does not exist, then we will
    performe the calibration first, store/save the calibration file and 
    return the calibration object (look above)
    """
    try:
        with open(CALIBRATION_PATH, "rb") as f:
                calibration = pickle.load(f)
    except OSError as err:
        print("OS error: {0}. Camera Calibration file wasn't found. We will proceed to create it".format(err))    
        calibration = calibrate_camera(CAL_IMAGE_PATH, ROWS, COLS)
        with open(CALIBRATION_PATH, 'wb') as f:
            pickle.dump(calibration, file=f)
    
        
    return calibration
