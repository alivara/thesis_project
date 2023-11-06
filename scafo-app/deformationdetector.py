import cv2
import numpy as np
# from skimage.metrics import structural_similarity
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from sklearn.cluster import KMeans
from sklearn import preprocessing as p
from collections import defaultdict
from matplotlib.pyplot import figure
from imagemanager import ImageManager
import imutils
import math 
import time

MAN = ImageManager()


class DeformationDetector:
    def __init__(self):
        # roi
        self.frame_pos_px = None
        self.frame_dist_cm = None
        self.frame_axis_cm = None
        self.frame_axis = None
        self.frame_position = None
        self.led_positions= None
        self.contors_ltr = None
        self.contors_rtl = None
        self.led_boxes = None
        self.positions_green = None
        self.fo_box = None
        self.pixel_increase = None
        # kmeans
        self.num_k = None
        # dipole
        self.n_dipole = None
        self.percentile_threshold = None
        self.thresh_multiplier = None
        # shifting tags
        self.shifting_tags_pos_px = None
        self.dipoles_base = None
        self.dipoles_mon = None
        self.dipoles_shiffted = None
        self.shift_img = None
        self.positions_blue = None
        # 1 cm to px ratios
        self.horizontal_ratio = 0 
        self.vertical_ratio = 0 

        # calibration bumps dimensions
        self.cal_bumps_height = None
        self.cal_bumps_diameter = None
        self.cal_bumps_volume = None
        self.cal_bumps_similarity = None
        self.jac_con = None

        # minimum bump area to be detected
        self.min_det_area_cm = 2 
        # minimum shifting to be detected
        self.min_shift_cm = 1
        # mimimum green ratio to consider crop as a shifting tag
        self.max_green_ratio = 0.1

        self.noise_T = 0
        self.blur_T = 0
        self.thresh_br = None
        # jaccard
        self.r_j_box = None
        # bumps calibration function (regression)
        self.reg_m = 0
        self.reg_b = 0

    def boxes_overlap(self, box1, box2): #https://www.geeksforgeeks.org/find-two-rectangles-overlap/
        '''
        removed due to licence
        '''
    
        return True

    def color_ranges(self, color):
        if color == 'blue':
            # lower = np.array([100, 200, 0])
            # upper = np.array([125, 255, 255])
            # --------------edited 
            lower = np.array([100, 210, 0])
            upper = np.array([125, 255, 255])

        elif color == 'green':
            lower = np.array([60, 150, 0])
            upper = np.array([89, 255, 255])
            # lower = np.array([40, 40, 40])
            # upper = np.array([80, 255, 255])
        elif color == 'red':
            lower = np.array([160, 0, 0])
            upper = np.array([179, 255, 255])
        elif color == 'white':
            sensitivity = 100
            lower = np.array([0,0,255-sensitivity])
            upper = np.array([255,sensitivity,255])
        elif color == 'plate_blue':
            # print("\n Blue for high light config")
            sensitivity = 5
            lower = np.array([0,0,255-sensitivity])
            upper = np.array([255,sensitivity,255])
        return lower, upper
    
    def remove_inner_boxes(self,contours):
        '''
        removed due to licence
        '''
        
        return contours

    def detect_led_tags(self, image, led_color, led_count):
        '''
        removed due to licence
        '''
        return led_positions, masked_output, with_contours 

    def clockwise_frame_positions(self, corners):
        # order led positions ABCD clockwise
        corners = np.array(corners)
        # the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
        led_sum = corners.sum(axis = 1)
        a = corners[np.argmin(led_sum)]
        c = corners[np.argmax(led_sum)]
        # the top-right point will have the smallest difference, whereas the bottom-left will have the largest difference
        led_diff = np.diff(corners, axis = 1)
        b = corners[np.argmin(led_diff)]
        d = corners[np.argmax(led_diff)]
        return (a, b, c, d)

    def get_frame_tags(self, image):
        
        led_positions, masked_output, with_contours = self.detect_led_tags(image, 'blue', None)
        led_positions = self.clockwise_frame_positions(led_positions)
        self.led_positions = np.array(led_positions)
        a = led_positions[0]
        b = led_positions[1]
        c = led_positions[2]
        d = led_positions[3]
        
        x_offset = int(self.pixel_increase / 2)
        y_offset = int(self.pixel_increase / 2)
        a, b, c, d = [a + [-x_offset, -y_offset], b + [x_offset, -y_offset],
                                  c + [x_offset, y_offset], d + [-x_offset, y_offset]]
        print(f'\n Modified Frame Position (increase: {self.pixel_increase}):')
        print(f'A: {a}, B: {b}, C: {c}, D: {d}')
        self.frame_pos_px = (a, b, c, d)
        
        return masked_output, with_contours

    

    def get_shifting_tags(self, image, store_positions=False):
        '''
        removed due to licence
        '''
        return led_positions, masked_output, with_contours

    def perspective_transform(self, image):
        a, b, c, d = self.frame_pos_px
        # compute the width of the new image, which will be the maximum distance between 
        widthA = np.sqrt(((c[0] - d[0]) ** 2) + ((c[1] - d[1]) ** 2)) # bottom-right and bottom-left x-coordiates 
        widthB = np.sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2)) # or top-right and top-left x-coordinates
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the maximum distance between
        heightA = np.sqrt(((b[0] - c[0]) ** 2) + ((b[1] - c[1]) ** 2)) # top-right and bottom-right y-coordinates
        heightB = np.sqrt(((a[0] - d[0]) ** 2) + ((a[1] - d[1]) ** 2)) # top-left and bottom-left y-coordinates
        maxHeight = max(int(heightA), int(heightB))

        dst_coordinates = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")
        src_coordinates = np.float32([a, b, c, d]) 
        # compute the perspective transform matrix and then apply it
        perspective_matrix = cv2.getPerspectiveTransform(src_coordinates, dst_coordinates)
        warped = cv2.warpPerspective(image, perspective_matrix, (maxWidth, maxHeight))
        
        return warped

    def combine_images(self, img1, img2):
        combined_image = img1.copy()
        i = 1
        alpha = 1.0/(i + 1)
        beta = 1.0 - alpha
        combined_image = cv2.addWeighted(img2, alpha, combined_image, beta, 0.0)
        return combined_image

    def detect_shifting(self, led_positions, base_img, shifted_img,homo_mat):
        '''
        removed due to licence
        '''

        return combined_img, shifting_array

    def threshold_evaluation(self, iref_img, i_img):
        # convert iref to grayscale
        iref_gray_img = cv2.cvtColor(iref_img, cv2.COLOR_BGR2GRAY) 

        # convert I to grayscale
        i_gray_img = cv2.cvtColor(i_img, cv2.COLOR_BGR2GRAY)

        # interpolate Iref and I to [-127,127]
        iref_gray_img_interp = np.interp(iref_gray_img, [0,255], [-127,127])
        i_gray_img_interp = np.interp(i_gray_img, [0,255], [-127,127])

        # compute the difference between I and Iref
        diff_img = np.absolute(i_gray_img_interp - iref_gray_img_interp) 

        # compute the histogram of Idiff
        vals = diff_img.flatten()
        fig = plt.figure()
        b, bins, patches = plt.hist(vals, 255)

        # get the mean and max values and plot them. What is T in that case??
        mean = vals.mean()
        max = vals.max()
        plt.axvline(mean, color='k', linestyle='dashed', linewidth=1)
        plt.axvline(max, color='r', linestyle='dashed', linewidth=1)

        # return the plot
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image_from_plot

    
    # def construct_morph_shapes(self, bump_diff_img):
    def construct_morph_shapes(self, bump_diff_img):
        img = bump_diff_img.copy()

        # here we can try to find an automatic method to detect the blur threshold        
        # we can compute by counting the number of bars / width of the cropped image
        # this could be a machine learning problem
        # bar_width = 3 

        # blur the image 
        # blur_ksize = (self.blur_T, 5) 
        # blur_ksize = (self.blur_T, self.blur_T)
        blur_ksize = (self.blur_T, 5) 
        # blur_ksize = (self.blur_T, self.blur_T) 
        img = cv2.blur(img, blur_ksize)

        # perform a series of erosions and dilations
        img = cv2.dilate(img, None, iterations = 4)
        img = cv2.erode(img, None, iterations = 4)

        # binarize the image
        img = cv2.threshold(img, 4, 255, cv2.THRESH_BINARY)[1]

        # construct a closing kernel and apply it to the thresholded image
        # closing_ksize = (50, 50) # sliding winow of pixels to be connected together
        closing_ksize = (50, 50)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, closing_ksize)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        return img

    def retrieve_color_ratio(self, img, color):
        '''
        removed due to licence
        '''
        return ratio
        
    def detect_bump_location(self, morph_img, iref_img, bump_img, ltr=True):
        '''
        removed due to licence
        '''
        return large_boxes
    
    def evaluate_jaccard(seld,cap_img, ref_img, log_evaluation=True):
        '''
        removed due to licence
        '''
        return 'Jac'
        
    def cluster(self,feature,n_clust):
        '''
        removed due to licence
        '''

        return z_p, label_center, min_max_clustes, mm_clus_f
    

    def dif_jac_val(self,cap_img1,ref_img1,tol_jc=0.01,log_evaluation=True):
        '''
        removed due to licence
        '''
        return image_from_plot
    
    def dipole_detector(self,img,mode):
        '''
        removed due to licence
        '''
        return 'dipole'

    def find_brightest_pixel_center(self,roi, percentile_value):
        bright_pixels = np.where(roi >= percentile_value)
        if bright_pixels[0].size == 0:
            return tuple(np.mean(bright_pixels, axis=1).astype(int))
        else:
            return tuple(np.mean(bright_pixels, axis=1).astype(int))

    def dipole_detector_02(self,img,mode): 
        '''
        removed due to licence
        '''
        return 'detection'
                  
        
    def dipole_shifting(self,base_img,mon_img,led_pos):
        '''
        removed due to licence
        '''
        
        MAN.store_image(combined_img, 'shift_img')
        
    
    def red_counter(self,img):
        ''' This function added to new version '''
        ## BGR method
        # RGB - Red
        r = img.copy()
        r[:, :, 0] = 0
        r[:, :, 1] = 0
        ret,thresh1 = cv2.threshold(r,150,255,cv2.THRESH_BINARY)
        thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_BGR2GRAY)
        red_px = cv2.countNonZero(thresh1)
        print(f'The number of red after TSH pixels is: {str(red_px)}')
        return int(red_px)

    def evaluate_similarity(self, box, bump_img, iref_img, binary_evaluation=False):
        '''
        removed due to licence
        '''

        return res

    def evaluate_bumps(self, large_boxes, iref_img, bump_img, reverse_evaluation=False, binary_evaluation=False):
        '''
        removed due to licence
        '''
        return correlations, with_contours

    def set_avg_correlations(self, corr_ltr, corr_rtl):
        both_directions = np.array([corr_ltr, corr_rtl])
        mean_values = np.mean(both_directions, axis=0)
        self.cal_bumps_similarity = mean_values

    def plot_regression(self, logscale=False):
        fig = plt.figure()
        x = np.array(self.cal_bumps_volume) * 0.001 # from cubic mm to cubic cm
        y = np.log(self.cal_bumps_similarity) if logscale else self.cal_bumps_similarity
        plt.scatter(x, y)

        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m*x + b)

        self.reg_m = m
        self.reg_b = b

        plt.grid()
        plt.xlabel("bump volume (cm^3)")
        # plt.xticks(x)
        plt.ylabel("similarity")

        # return the plot
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image_from_plot
