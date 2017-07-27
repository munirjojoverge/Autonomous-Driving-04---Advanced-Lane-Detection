'''
##__________________________________________________________________________
##                              IMAGE PROCESSING UTILS
##
## This file defines ALL the functions described on
## the Notebook used as a writeup. "Advanced-Lane-Finding-Submission.ipynb"
## This notebook should be use as a refernce for this file.
## There are also some other functions implemented to support the Lane Detection
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
import cv2
import numpy as np
from scipy import signal

# Define a function to threshold (binary) a specific channel (you pass, for example, the s channel = hls[:,:,2] and the theshold values)
def binary_thresh(img_ch, thresh=(0, 100)):    
    binary_output = np.zeros_like(img_ch)
    binary_output[(img_ch > thresh[0]) & (img_ch <= thresh[1])] = 1    
    # Return the binary image
    return binary_output

# Define a function to threshold (just therhold and not convert to binary) a specific channel (you pass, for example, the s channel = hls[:,:,2] and the theshold values)
def color_thresh(img_ch, thresh=(0, 100)):    
    binary_output = binary_thresh(img_ch, thresh)    
    filtered_img = binary_output * img_ch
    # Return a color image
    return filtered_img

# Define a function that applies Gaussian smoothing bluring to and image (1 to 3 channles)
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Define a function that takes an image (alredy converted into grayscale - to avoid not applying the right conversion -, 
# gradient orientation (x or y), the sobel kernel (max 31, min 3, only odd numbers) and threshold (min, max values).
def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # Create the binaty filtered image
    binary_output = binary_thresh(scaled_sobel, thresh=thresh) 

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values.
# as before, the img passed should be already in grayscale to avoid not applying the right conversion
# This is exactly the same as cv2.laplace but we can specify the kernel in this case
def mag_thresh(gray, sobel_kernel=3, thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    
    # Create the binaty filtered image
    binary_output = binary_thresh(gradmag, thresh=thresh)

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_thresh(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    
    # Create the binaty filtered image
    binary_output = binary_thresh(absgraddir, thresh=thresh) 

    # Return the binary image
    return binary_output



def colorFilter(image, colorBoundaries, blur=1):
    img = gaussian_blur(image, blur)
    # loop over the boundaries
    for (lower, upper) in colorBoundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        #output = cv2.bitwise_and(image, image, mask = mask)
               
        return mask
    

def generate_lane_mask(img, y_cutoff=0):
    """
    Generates a binary mask selecting the lane lines of an street scene image.
    Input Parameters:
        img: RGB color image
        y_cutoff: vertical cutoff to limit the search area (pixels on the Y direction cut from the top)
    Output:
        binary mask
    """
    rgb = img[y_cutoff:, :, :]
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
    yuv = 255 - yuv
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    chs = np.stack((yuv[:, :, 0], yuv[:, :, 1], hls[:, :, 2]), axis=2)
    gray = np.mean(chs, 2)

    s_x = abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(5,255))
    s_y = abs_sobel_thresh(gray, orient='y', sobel_kernel=3, thresh=(0,255))

    grad_dir = dir_thresh(gray, sobel_kernel=3, thresh=(0.65,1.05))
    grad_mag = mag_thresh(gray, sobel_kernel=3, thresh=(200,255))

    # Extract Yellow as we did on the Notebook (over the HLS image)
    YellowBoundary = [([20, 50, 150], [40, 255, 255])]
    Yellow_Highlights = colorFilter(hls, YellowBoundary)
    
    # Extract White as we did on the Notebook (over the RBG image)
    WhiteBoundary = [([175, 150, 200], [255, 255, 255])]
    White_Highlights = colorFilter(rgb, WhiteBoundary)
    
    # Extract higlights by looking at the Red channel as we did on the Notebook (over the RBG image)
    RedChBoundary = [(50,255)]
    RedCh_Highlights = extract_highlights(rgb[:, :, 0],99.0)
    
    mask = np.zeros(img.shape[:-1], dtype=np.uint8)

    mask[y_cutoff:, :][((s_x == 255) & (s_y == 255)) |
                       ((grad_mag == 255) & (grad_dir == 255)) |
                       ((Yellow_Highlights == 255) | (White_Highlights == 255) | (RedCh_Highlights == 255))] = 1

    mask = binary_noise_reduction(mask, 4)
    
    return mask

def extract_highlights(img, p=99.9):
    """
    Generates an image mask selecting highlights.
    Input Parameters:
        img: image with pixels in range 0-255
        p: percentile for highlight selection. default=99.9
        
    :return: Highlight 255 not highlight 0
    """
    p = int(np.percentile(img, p) - 30)
    mask = cv2.inRange(img, p, 255)
    return mask

def binary_noise_reduction(img, thresh):
    """
    Reduces noise of a binary image by applying a filter which counts neighbours with a value
    and only keeping those which are above the threshold.
    :param img: binary image (0 or 1)
    :param thresh: min number of neighbours with value
    :return:
    """
    k = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    nb_neighbours = cv2.filter2D(img, ddepth=-1, kernel=k)
    img[nb_neighbours < thresh] = 0
    return img


def histogram_lane_detection(img, steps, search_window, h_window):
    """
    Detects lane lines by applying a sliding histogram.
    Input Parameters:
        img: binary image
        steps: steps for the sliding histogram windows
        search_window: Tuple which limits the horizontal search space.
        h_window: window size for horizontal histogram smoothing
    
    :return: x, y of detected pixels (Line detected)
    """
    all_x = []
    all_y = []
    masked_img = img[:, search_window[0]:search_window[1]]
    pixels_per_step = img.shape[0] // steps

    for i in range(steps):
        start = masked_img.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step
        histogram = np.sum(masked_img[end:start, :], axis=0)
        histogram_smooth = signal.medfilt(histogram, h_window)
        peaks = np.array(signal.find_peaks_cwt(histogram_smooth, np.arange(1, 5)))

        highest_peak = highest_n_peaks(histogram_smooth, peaks, n=1, threshold=5)
        if len(highest_peak) == 1:
            highest_peak = highest_peak[0]
            center = (start + end) // 2
            x, y = get_pixel_in_window(masked_img, highest_peak, center, pixels_per_step)

            all_x.extend(x)
            all_y.extend(y)

    all_x = np.array(all_x) + search_window[0]
    all_y = np.array(all_y)

    return all_x, all_y


def highest_n_peaks(histogram, peaks, n=2, threshold=0):
    """
    Returns the n highest peaks of a histogram above a given threshold.
    Input Parameters:
        histogram:
        peaks: list of peak indexes
        n: number of peaks to select
        threshold:
    
    :return: n highest peaks
    
    """
    if len(peaks) == 0:
        return []

    peak_list = [(peak, histogram[peak]) for peak in peaks if histogram[peak] > threshold]
    peak_list = sorted(peak_list, key=lambda x: x[1], reverse=True)

    if len(peak_list) == 0:
        return []

    x, y = zip(*peak_list)
    x = list(x)

    if len(peak_list) < n:
        return x

    return x[:n]


def detect_lane_along_poly(img, poly, steps):
    """
    Slides a window along a polynomial an selects all pixels inside.
    Input Parameters:
        img: binary image
        poly: polynomial to follow
        steps: number of steps for the sliding window
    
    :return: x, y of detected pixels
    """
    pixels_per_step = img.shape[0] // steps
    all_x = []
    all_y = []

    for i in range(steps):
        start = img.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step

        center = (start + end) // 2
        x = poly(center)

        x, y = get_pixel_in_window(img, x, center, pixels_per_step)

        all_x.extend(x)
        all_y.extend(y)

    return all_x, all_y


def get_pixel_in_window(img, x_center, y_center, size):
    """
    returns selected pixel inside a window.
    Input Parameters:
        img: binary image
        x_center: x coordinate of the window center
        y_center: y coordinate of the window center
        size: size of the window in pixel
    
    :return: x, y of detected pixels
    """
    half_size = size // 2
    window = img[y_center - half_size:y_center + half_size, x_center - half_size:x_center + half_size]

    x, y = (window.T == 1).nonzero()

    x = x + x_center - half_size
    y = y + y_center - half_size

    return x, y


def calculate_lane_area(lanes, area_height, steps):
    """
    Returns a list of pixel coordinates marking the area between two lanes
    :param lanes: Tuple of Lines. Expects the line polynomials to be a function of y.
    :param area_height:
    :param steps:
    :return:
    """
    points_left = np.zeros((steps + 1, 2))
    points_right = np.zeros((steps + 1, 2))

    for i in range(steps + 1):
        pixels_per_step = area_height // steps
        start = area_height - i * pixels_per_step

        points_left[i] = [lanes[0].best_fit_poly(start), start]
        points_right[i] = [lanes[1].best_fit_poly(start), start]

    return np.concatenate((points_left, points_right[::-1]), axis=0)


def IsLane(line_one, line_two, parallel_thresh=(0.0003, 0.55), dist_thresh=(350, 460)):
    """
    Checks if two lines are likely to form a lane by comparing the curvature and distance.
    Basically are they parallel and if so, is the distance between them a reasonable Lane size
    
    Input Parameters:
        line_one: Class Line object (look Line.py)
        line_two: Class Line object (look Line.py)
        parallel_thresh: Tuple of float values representing the delta threshold for the
                         first and second coefficient of the polynomials. Please refer to the 2 order polinomial approximation we used
        dist_thresh: Tuple of integer values marking the lower and upper threshold
                     for the distance between the lines of a lane.
    return:
        Boolean
    """
    is_parallel = line_one.is_parallel_to(line_two, threshold=parallel_thresh)
    dist = line_one.get_current_fit_distance(line_two)
    is_within_dist = dist_thresh[0] < dist < dist_thresh[1]

    return is_parallel & is_within_dist



def draw_poly(img, poly, steps, color, thickness=10, dashed=False):
    """
    Draws a polynomial onto an image.
    Input Parameters:
        img:
        poly:
        steps:
        color:
        thickness:
        dashed:

    return:
        image (same shape as img) with the polinomio drawn
    """
    
    img_height = img.shape[0]
    pixels_per_step = img_height // steps
    
    for i in range(steps):
        start = i * pixels_per_step
        end = start + pixels_per_step

        start_point = (int(poly(start)), start)
        end_point = (int(poly(end)), end)

        if dashed is False or i % 2 == 1:
            img = cv2.line(img, end_point, start_point, color, thickness)

    return img
    
    
def draw_line(img, pts, color, thickness=10):    
    return cv2.polylines(img, np.int_([pts]), isClosed=False, color=color, thickness = thickness)
    

def outlier_removal(x, y, q=5):
    """
    Removes horizontal outliers based on a given percentile.
    Input Parameters:
        x: x coordinates of pixels
        y: y coordinates of pixels
        q: percentile
    
    :return: cleaned coordinates (x, y)
    """
    if len(x) == 0 or len(y) == 0:
        return x, y

    x = np.array(x)
    y = np.array(y)

    lower_bound = np.percentile(x, q)
    upper_bound = np.percentile(x, 100 - q)
    selection = (x >= lower_bound) & (x <= upper_bound)
    return x[selection], y[selection]
