'''
##__________________________________________________________________________
##                              LANE LINE CLASS DEFINITION
##
## This file defines the LaneLine Class that will try to "fit" a line found on the image. 
## I investigated and tried several features to improve the performance of "fitting".
## Definetely this needs more work but is a good start.
## Some of the main ideas were discussed in the lectures and on the notebook used as a writeup. 
## In this notebook should be use as a refernce for this file.
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
import numpy as np
import math

class LaneLine:
    ## PROPERTIES & CONSTRUCTOR
    def __init__(self, n_frames=1, x=None, y=None):
        """
        Define a class to receive the characteristics of each line detection
            n_frames: Number of frames for smoothing
            x: initial x coordinates
            y: initial y coordinates
        """
        # Frame memory
        self.n_frames = n_frames
        # was the line detected in the last iteration?
        self.detected = False
        # number of pixels added per frame
        self.n_pixel_per_frame = []
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = None
        # Polynom for the current coefficients
        self.current_fit_poly = None
        # Polynom for the average coefficients over the last n iterations
        self.best_fit_poly = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        if x is not None:
            self.update(x, y)

    def update(self, x, y):
        """
        Updates the line representation.
        Input Parameters
            x: list of x values        
            y: list of y values
        """
        assert len(x) == len(y), 'x and y have to be the same size'

        self.allx = x
        self.ally = y

        self.n_pixel_per_frame.append(len(self.allx))
        self.recent_xfitted.extend(self.allx)

        if len(self.n_pixel_per_frame) > self.n_frames:
            n_x_to_remove = self.n_pixel_per_frame.pop(0)
            self.recent_xfitted = self.recent_xfitted[n_x_to_remove:]

        self.bestx = np.mean(self.recent_xfitted)

        self.current_fit = np.polyfit(self.allx, self.ally, 2)

        if self.best_fit is None:
            self.best_fit = self.current_fit
        else:
            self.best_fit = (self.best_fit * (self.n_frames - 1) + self.current_fit) / self.n_frames

        self.current_fit_poly = np.poly1d(self.current_fit)
        self.best_fit_poly = np.poly1d(self.best_fit)

    def is_parallel_to(self, other_line, threshold=(0, 0)):
        """
        Checks if two lines are parallel by comparing their first two coefficients.
        :param other_line: Line to compare to
        :param threshold: Tuple of float values representing the delta thresholds for the coefficients.
        :return:
        """
        first_coefi_dif = np.abs(self.current_fit[0] - other_line.current_fit[0])
        second_coefi_dif = np.abs(self.current_fit[1] - other_line.current_fit[1])

        is_parallel = first_coefi_dif < threshold[0] and second_coefi_dif < threshold[1]

        return is_parallel

    def get_current_fit_distance(self, other_line):
        """
        Gets the distance between the current fit polynomials of two lines
        :param other_line:
        :return:
        """
        return np.abs(self.current_fit_poly(719) - other_line.current_fit_poly(719))

    def get_best_fit_distance(self, other_line):
        """
        Gets the distance between the best fit polynomials of two lines
        :param other_line:
        :return:
        """
        return np.abs(self.best_fit_poly(719) - other_line.best_fit_poly(719))


def calc_curvature(curve):
    """
    Calculates the curvature of a line in meters
    Input parameters:
        curve: this is the set of points on the center of the lane
    Return:
        radius of curvature in meters
    """

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension

    y = np.array(np.linspace(0, 719, num=10))
    x = np.array([curve(x) for x in y])
    y_eval = np.max(y)

    fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
    curverad = ((1 + (2 * fit_cr[0] * y_eval / 2. + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

    return curverad

def calc_desiredSpeed(roc):
    """
    Calculates the desired speed in meters/sec (on acurve) based on some studies that show that the
    threshold value of comfort is 1.8 m/s2, with medium comfort and discomfort levels of 3.6 m/s2 and 5 m/s2, respectively
    "W. J. Cheng, Study on the evaluation method of highway alignment comfortableness [M.S. thesis], 
    Hebei University of Technology, Tianjin, China, 2007."
    Input Parameters:
        roc: Radious Of Curvature (m)
        
    return: Desired Speed in meters/s
    """
    speedLimit = 33.3333 # (120.0 km/h)
    
    v = math.sqrt(1.8 * roc)
    if v >= speedLimit:
        v = speedLimit
    
    return v