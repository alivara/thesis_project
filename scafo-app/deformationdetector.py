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
        l1 = (box1[0], box1[1]) #x, y
        r1 = (box1[0]+box1[2], box1[1]+box1[3])

        l2 = (box2[0], box2[1])
        r2 = (box2[0]+box2[2], box1[1]+box2[3])

        # To check if either rectangle is actually a line
        # For example  :  l1 ={-1,0}  r1={1,1}  l2={0,-1}  r2={0,1}
        
        if (l1[0] == r1[0] or l1[1] == r1[1] or l2[0] == r2[0] or l2[1] == r2[1]):
            # the line cannot have positive overlap
            return False
        
        # If one rectangle is on left side of other
        if(l1[0] >= r2[0] or l2[0] >= r1[0]):
            return False
    
        # If one rectangle is above other
        if(r1[1] >= l2[1] or r2[1] >= l1[1]):
            return False
    
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
        # Sort contours in descending order of their areas
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        to_remove = []
        
        for i in range(len(contours)):
            for j in range(i+1, len(contours)):
                # Get the bounding rectangle of the j-th contour
                x_j, y_j, w_j, h_j = cv2.boundingRect(contours[j])
                # Check if all four corners of the bounding rectangle are inside the i-th contour
                if all(cv2.pointPolygonTest(contours[i], (x_j+k, y_j+l), False) == 1 for k in range(w_j+1) for l in range(h_j+1)):
                    to_remove.append(j)
        
        # Remove the inner contours
        contours = [contours[i] for i in range(len(contours)) if i not in to_remove]
        
        return contours

    def detect_led_tags(self, image, led_color, led_count):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale 
        gray = cv2.merge([gray, gray, gray]) # obtain 3 channel grayscale image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # convert to hsv

        lower, upper = self.color_ranges(led_color)

        mask = cv2.inRange(hsv, lower, upper)
        masked_output = cv2.bitwise_and(image, image, mask=mask) # apply hsv filtering to the image
        masked_output_gray = cv2.cvtColor(masked_output, cv2.COLOR_BGR2GRAY)

        thresh, binary = cv2.threshold(masked_output_gray, 0, 255, cv2.THRESH_BINARY)

        # find contours and draw a bounding box around all of them on the binary image
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_TREE
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Remove inner contours
        contours = self.remove_inner_boxes(contours)

        with_contours = binary.copy()

        detected_boxes=[]
        # draw only the first x biggest contours
        led_positions = []
        for c in contours:
            box = cv2.boundingRect(c)
            x, y, w, h = box
            
            # Make sure contour area is large enough to draw a rect around it
            if (cv2.contourArea(c)) == 0:
                continue

            # Make sure that the next box does not overlap with any added box
            for db in detected_boxes:
                overlaps = self.boxes_overlap(box, db)
                if overlaps:
                    continue

            cv2.rectangle(with_contours,(x,y), (x+w,y+h), (255,255,255), 5) #the color of the box will be white always since the image is grayscale
            detected_boxes.append(box)
            led_positions.append([x+w//2, y+h//2])
            
            # remove this based on the not known number of LEDs
            # if len(led_positions) == led_count:
            #     break

        # Find the contour with the largest area
        self.fo_box = max(contours, key=cv2.contourArea)

        # # Find the minimum enclosing rectangle for the largest contour
        x, y, w, h = cv2.boundingRect(self.fo_box)
        cv2.rectangle(with_contours,(x,y), (x+w+50,y+h+50), (255,255,255), 10)
        self.led_boxes = detected_boxes
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
        """ Use raw images not the ones resulting from perspective transform """
        led_positions, masked_output, with_contours = self.detect_led_tags(image, 'green', 4)
        led_positions = self.clockwise_frame_positions(led_positions)

        if store_positions:
            self.shifting_tags_pos_px = led_positions
            print('\nInitial shifting tags positions:')
            print(f'A: {led_positions[0]}, B: {led_positions[1]}, C: {led_positions[2]}, D: {led_positions[3]}')

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
        Update this function for using the homography matrix in calculation.
        Check the updated algorithm in real test to make sure about the working progress.
        '''
        n = len(led_positions)
        shifting_array = np.zeros(n)
        combined_img = self.combine_images(base_img, shifted_img)
        for i in range(n):
            before = self.shifting_tags_pos_px[i]
            after = led_positions[i]
            
            
            # getting the pixel diffrence
            before_dmm = np.array([before[0], before[1]])
            after_dmm = np.array([after[0], after[1]])
            
            # L2 norm
            shifting_array[i] = round(np.linalg.norm(before_dmm-after_dmm), 2)
            if shifting_array[i] > self.min_shift_cm: # draw arrow if larger than 1 cm
                print(f'Detected shifting of led {str(i+1)} by {shifting_array[i]} px')
                combined_img = cv2.arrowedLine(combined_img, before, after, (0, 0, 255), 9)

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
        # apply hsv mask
        lower, upper = self.color_ranges(color)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        mask = cv2.inRange(hsv, lower, upper)
        masked_output = cv2.bitwise_and(img, img, mask=mask)

        # convert to grayscale
        masked_output_gray = cv2.cvtColor(masked_output, cv2.COLOR_BGR2GRAY)

        # get all non black Pixels from grayscale
        non_black = cv2.countNonZero(masked_output_gray)

        # get ratio of non black pixels
        total_pixels = masked_output_gray.shape[0] * masked_output_gray.shape[1]
        ratio = non_black / total_pixels
        return ratio
        
    def detect_bump_location(self, morph_img, iref_img, bump_img, ltr=True):
        # find the contours in the binary image, then sort the contours
        img = morph_img.copy().astype(np.uint8)
        
        ## cv2.cvtColor added to convert image unit type for find 
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)

        large_boxes = []
        combined_img = self.combine_images(iref_img, bump_img) 
        for c in contours:
            # use straight rect (cv.boundingRect) instead of cv.minAreaRect which is rotated 
            x, y, w, h = cv2.boundingRect(c)
            
            # filter out small boxes (lower than min_det_area_cm)
            if w < (self.min_det_area_cm * self.horizontal_ratio) or h < (self.min_det_area_cm * self.vertical_ratio):
                continue

            # filter out greenish boxes because those will be related to shifting leds
            crop = combined_img[y:y+h, x:x+w]
            ratio = self.retrieve_color_ratio(crop, 'green')
            if ratio > self.max_green_ratio:
                print('greenish bump: to be skipped')
                continue

            large_boxes.append((x, y, w, h))

        # order rtl or ltr (only compare on the x axis, all the bumps should be aligned)
        large_boxes = sorted(large_boxes, key=lambda t: (t[0]+t[2])/2, reverse=not ltr)

        return large_boxes
    #*************************************************************************
    def evaluate_jaccard(seld,cap_img, ref_img, log_evaluation=True):
        ''' This function added to new version '''
        intersection = np.minimum(cap_img, ref_img)
        #intersection = np.multiply(bump_crop, original_crop)
        union = np.maximum(cap_img, ref_img)
        res = np.sum(intersection) / np.sum(union)
        # check this part with logaritmic scale
        
        return res
        # if log_evaluation:
        #     return math.log(res)
        # else:
        #     if 1-res >= 0.1:
        #         return 0.1
        #     else:
        #         return (1-res)

    def cluster(self,feature,n_clust):
        kmeans = KMeans(init="k-means++", n_clusters=n_clust, n_init=10, random_state=0).fit(feature) # running Kmeans with n features
        z_p = kmeans.predict(feature)

        # find min and max of clusters
        mm_clus = defaultdict(list)
        label_value = [[z_p[i],feature[i][0]] for i in range(0,feature.shape[0])]
        for k in label_value:
            mm_clus[k[0]].append(k[1])  #creating dictionary containing first element of arr as key and last element as value
        mm_clus = dict(mm_clus)

        mm_clus_f = defaultdict(list)
        min_max_clustes = list()
        for k1,v1 in mm_clus.items():
            mm_clus_f[k1].append([min(v1), max(v1)])
            min_max_clustes.extend((min(v1), max(v1)))
        mm_clus_f = dict(mm_clus_f)
        list(set(min_max_clustes)).sort()

        # lebales and centers of kmeans
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        label_center = [[list(set(labels))[i],centers[i,0]] for i in range(0,centers.shape[0])]

        return z_p, label_center, min_max_clustes, mm_clus_f
    

    def dif_jac_val(self,cap_img1,ref_img1,tol_jc=0.01,log_evaluation=True):
        '''
        This function is for caculating the difference with jaccad.
        '''
        cap_img = cv2.GaussianBlur(cap_img1,(0,0),sigmaX=16)
        ref_img = cv2.GaussianBlur(ref_img1,(0,0),sigmaX=16)        
    
        # convert grayscale
        cap_img = cv2.cvtColor(cap_img, cv2.COLOR_BGR2GRAY)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

        ## define the similarity and x and y
        cube_size = 52 # define the pixel of each square
        step = 20 # detine the step of moving
        s_point = int(cube_size/2) # define the start and stop point
        t1 = time.time()
        y = [i for i in range(s_point,(ref_img.shape[0]-s_point),step)] # define the y axis of plot
        x = [i for i in range(s_point,(ref_img.shape[1]-s_point),step)] # define the x axis of plot
        z = [self.evaluate_jaccard(cap_img[i-s_point:i+s_point,j-s_point:j+s_point], 
                ref_img[i-s_point:i+s_point,j-s_point:j+s_point],log_evaluation=False) 
                        for i in y for j in x] # define the similarity on z axis
        print(f'-- sigma: 16 -- step: {step} -- time: {time.time()-t1} -- ')

        self.jac_con = math.isclose(np.mean(z),1,rel_tol=tol_jc)
        
        if log_evaluation:
            z = [math.log(num) for num in z]
            # z = math.log(z)
        
        X, Y = np.meshgrid(np.array(x),np.array(y))
        z = np.array(z)
        min_max_scaler = p.MinMaxScaler()
        z_norm = z.reshape(-1,1)
        # normilize z [0-100]
        z_norm = min_max_scaler.fit_transform(z_norm)*100
        Z = z_norm.reshape(len(y),len(x))
        # Clusterring the log(Jaccard)
        z_p, label_center, min_max_clustes, mm_clus_f = self.cluster(feature = z_norm,n_clust = self.num_k)

        # finding the min and max and the range for the color bar and visualization
        label_center.sort(key= lambda X: X[1],reverse= True)
        min_max_clustes.sort()
        label_final_cluster = [format((min_max_clustes[i]+min_max_clustes[i+1])/2,'.4') for i in range(1,len(min_max_clustes)-1,2)]
        label_final_cluster.append(format(max(min_max_clustes),'.4'))
        label_final_cluster.append(format(min(min_max_clustes),'.4'))
        label_final_cluster = [float(i) for i in label_final_cluster]
        label_final_cluster.sort()
        
        # plot config
        fig, ax = plt.subplots(2)
        fig.set_figheight(20)
        fig.set_figwidth(20)
        
        # plot Jaccard:
        c1 = ax[0].pcolormesh(X.T, Y.T,Z.T, cmap='gray')
        ax[0].invert_yaxis()
        ax[0].set_title('Correlation of normalized log logarithm Jaccard index (LEDs are in Blue)')
        fig.colorbar(c1, ax=ax[0])

        # plt_kmeans:
        color_list_mat = ['blue','purple','brown','red','green','gray','olive','orange','pink','cyan']
        cmap, norm = mcolors.from_levels_and_colors(levels=label_final_cluster, colors=color_list_mat[:len(label_final_cluster)-1]) #colors=['dimgray', 'brown', 'blue','green']
        c2 = ax[1].pcolormesh(X.T, Y.T,Z.T, cmap=cmap, norm=norm)
        ax[1].invert_yaxis()
        ax[1].set_title('Clustered data (LEDs are in Black)')
        fig.colorbar(c2, ax=ax[1])

        # showing the place of LEDs in Black
        for box in self.r_j_box:
            x_b, y_b, w_b, h_b = box
             # Create a Rectangle patch
            rect = patches.Rectangle((x_b, y_b), w_b, h_b,color = 'black') #  linewidth=1, edgecolor='r', facecolor='none'
            rect1 = patches.Rectangle((x_b, y_b), w_b, h_b, color = 'black')
            # Add the patch to the Axes
            ax[0].add_patch(rect)
            ax[1].add_patch(rect1)
        
        # return the plot
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        MAN.store_image(image_from_plot, 'jac_plot') 
        # return image_from_plot
    
    def dipole_detector(self,img,mode):
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # self.get_frame_tags
        if mode == 'base':
            color_contours = [['blue',None,[]],['green',None,[]]]
        else:
            color_contours = [['blue',None,[]]]
            
        for col in color_contours:
            # Define the range color in HSV
            lower, upper = self.color_ranges(col[0]) # color in HSV
            # Threshold the HSV image to get only blue and green colors
            mask_blue = cv2.inRange(hsv, lower, upper)
            masked_output = cv2.bitwise_and(img, img, mask=mask_blue)
            masked_output_gray = cv2.cvtColor(masked_output, cv2.COLOR_BGR2GRAY)
            
            thresh, binary = cv2.threshold(masked_output_gray, 0, 255, cv2.THRESH_BINARY)
            binary = cv2.medianBlur(binary, 5)
            col[1], _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort the contours by area in descending order
            col[1] = sorted(col[1], key=cv2.contourArea, reverse=True)

        n = self.n_dipole
        # n = 11
        # draw contours 
        all_box = []
        for col_cont in color_contours:
            detected_boxes=[]
            led_positions = []
            # Remove inner contours
            col_cont[1] = self.remove_inner_boxes(col_cont[1])
            # for cnt in col_cont[1]:

            for cnt in col_cont[1][:n]:
                box = cv2.boundingRect(cnt)
                x, y, w, h = box

                roi = gray[y:y + h, x:x + w]
                # brightest_pixel_value = np.max(roi)
                brightest_pixel_coords = np.unravel_index(np.argmax(roi), roi.shape)
                
                # Draw rectangle around the bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw circle around the brightest pixel
                brightest_pixel_center = (x + brightest_pixel_coords[1], y + brightest_pixel_coords[0])
                cv2.circle(img, brightest_pixel_center, 5, (0, 0, 255), -1)
                
                # Make sure contour area is large enough to draw a rect around it
                if (cv2.contourArea(cnt)) == 0:
                    continue
                # Make sure that the next box does not overlap with any added box
                for db in detected_boxes:
                    overlaps = self.boxes_overlap(box, db)
                    if overlaps:
                        continue
                
                if mode == 'base':
                    cv2.rectangle(img,(x,y), (x+w,y+h), (255,255,255), 3) #the color of the box will be white always since the image is grayscale
                
                if mode == 'base':
                    all_box.append(box)

                detected_boxes.append(box)
                led_positions.append([brightest_pixel_center[0],brightest_pixel_center[1]])
                # seperate colors
                if col_cont[0] == 'blue':
                    color_contours[0][2].append((int(brightest_pixel_center[0]),int(brightest_pixel_center[1])))
                if col_cont[0] == 'green':
                    color_contours[1][2].append((int(brightest_pixel_center[0]),int(brightest_pixel_center[1])))
        
        # Sort the blue and green dipoles by their y-coordinates
        color_contours[0][2].sort(key=lambda x: x[1])
        self.positions_blue = color_contours[0][2]
        
        # Find the closest blue and green positions and store them in a list
        if mode == 'base':  
            self.r_j_box = all_box
            color_contours[1][2].sort(key=lambda x: x[1])
            output_img = img.copy()
            dipoles = {}
            for i, pos_blue in enumerate(color_contours[0][2]):
                closest_distance = np.inf
                closest_pos_green = None
                for pos_green in color_contours[1][2]:
                    distance = np.linalg.norm(np.array(pos_blue) - np.array(pos_green))
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_pos_green = pos_green
                dipoles[i+1] = (pos_blue, closest_pos_green, closest_distance)
                # Draw line of the dipole 
                cv2.line(output_img, pos_blue, closest_pos_green, (0, 0, 255), 2)
                # Draw the number of the dipole inside the box
                dipole_pos = (pos_blue[0], closest_pos_green[1])
                cv2.putText(output_img, str(i+1), dipole_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2 + 1)

            # printing the dipole data        
            for key, value in dipoles.items():
                print(f"Dipole No.{key} -- Blue LED pos:{value[0]} -- Green LED pos:{value[1]} -- dist(px):{value[2]:.2f}")
            
            print('dipole')
            print(dipoles)
            print('aa')
            self.dipoles_base = dipoles
            MAN.store_image(output_img, 'base_dipole_img') 

    def find_brightest_pixel_center(self,roi, percentile_value):
        bright_pixels = np.where(roi >= percentile_value)
        if bright_pixels[0].size == 0:
            return tuple(np.mean(bright_pixels, axis=1).astype(int))
        else:
            return tuple(np.mean(bright_pixels, axis=1).astype(int))

    def dipole_detector_02(self,img,mode): 
        '''
        ,thresh_multiplier=0.7,percentile_threshold=0.3
        '''
        percentile_threshold = self.percentile_threshold
        thresh_multiplier = self.thresh_multiplier
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(grey_img)

        origin = np.array([0, 0])  # Origin point
        thresh = int( maxVal * thresh_multiplier)
        
        thresh, binary = cv2.threshold(grey_img , thresh, 255, cv2.THRESH_BINARY)
        # Want to get the number of points now
        contours, _  = cv2.findContours(binary, 
                                    cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE,
                                    offset=(0, 0))

        # Sort the contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        n = 2*(self.n_dipole)
        cont_n = contours[:2*(self.n_dipole)]
        regions = []
        for contour in cont_n:
            box = cv2.boundingRect(contour)
            x, y, w, h = box
            regions.append((x, y, w, h))
            color = (0,0,255,0)
            cv2.rectangle(img, (x, y), (x+w,y+h), color, 2)
        
        self.r_j_box = regions
        # Iterate through each contour and find its closest neighbor
        output_img = img.copy()
        dipoles = {}
        nd = 0

        for i, contour1 in enumerate(cont_n):
            min_distance = float('inf')
            closest_contour = None

            box = cv2.boundingRect(contour1)
            x, y, w, h = box
            roi1 = grey_img[y:y + h, x:x + w]
            # percentile_value = np.percentile(roi1, percentile_threshold)  # Get the percentile threshold value
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(roi1)
            percentile_value = minVal + (maxVal-minVal)*percentile_threshold
            center1 = self.find_brightest_pixel_center(roi1, percentile_value)
            center1 = np.array((x + center1[1], y + center1[0]))
            # center1 = np.mean(contour1, axis=0)[0] # old
            
            for j, contour2 in enumerate(cont_n):
                if i != j:
                    box = cv2.boundingRect(contour2)
                    x, y, w, h = box
                    roi2 = grey_img[y:y + h, x:x + w]
                    # percentile_value = np.percentile(roi2, percentile_threshold)
                    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(roi2)
                    percentile_value = minVal + (maxVal-minVal)*percentile_threshold
                    center2 = self.find_brightest_pixel_center(roi2, percentile_value)
                    center2 = np.array((x + center2[1], y + center2[0]))
                    # center2 = np.mean(contour2, axis=0)[0] # old
                    distance = np.linalg.norm(center1 - center2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_contour = contour2
            
            
            if closest_contour is not None:
                box = cv2.boundingRect(closest_contour)
                x, y, w, h = box
                roi2 = grey_img[y:y + h, x:x + w]
                # percentile_value = np.percentile(roi2, percentile_threshold)
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(roi2)
                percentile_value = minVal + (maxVal-minVal)*percentile_threshold
                center2 = self.find_brightest_pixel_center(roi2, percentile_value)
                center2 = np.array((x + center2[1], y + center2[0]))
                # center2 = np.mean(closest_contour, axis=0)[0] # old--------- top new
                
                # Convert center coordinates to integers
                dipole_contours = [center1, center2]
                dipole_contours.sort(key=lambda c: np.linalg.norm(origin - c)) # sort from center
                center1 = tuple(map(int, dipole_contours[0]))
                center2 = tuple(map(int, dipole_contours[1]))
                # Draw a line between the centers of the two contours
                if (center1, center2, min_distance) not in dipoles.values():
                    dipoles[nd+1] = (center1, center2, min_distance)
                    nd+=1
                    cv2.line(output_img, center1, center2, (0, 0, 255), 2)
                    dipole_pos = (center1[0], center2[1])
                    cv2.putText(output_img, str(nd), dipole_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2 + 1)
        
        # printing the dipole data        
        for key, value in dipoles.items():
            print(f"Dipole No.{key} -- T LED pos:{value[0]} -- B LED pos:{value[1]} -- dist(px):{value[2]:.2f}")


        
        if mode == 'base':  
            self.dipoles_base = dipoles
            MAN.store_image(output_img, 'base_dipole_img') 
        elif mode == 'mon':  
            self.dipoles_mon = dipoles    
                  
        
    def dipole_shifting(self,base_img,mon_img,led_pos):
        combined_img = self.combine_images(base_img, mon_img)
        # calculate dipoles distance, based on the closest points ro green
        dipoles_shifting = {}
        for key_base, value_base in self.dipoles_base.items():
            b_pos, g_pos, dis = value_base
            closest_distance = np.inf
            closest_pos_green = None
            # for pos in led_pos:
                # b_pos_mon = pos
            for key_mon, value_mon in self.dipoles_mon.items():
                b_pos_mon, g_pos_mon, dis_mon = value_mon
                distance = np.linalg.norm(np.array(b_pos) - np.array(b_pos_mon))
                if distance < closest_distance:
                    closest_distance = distance
                    closest_pos_green = b_pos_mon
            dipoles_shifting[key_base] = (b_pos, closest_pos_green, (closest_distance/dis)*50)
            # Draw line of the shifting dipole 
            cv2.line(combined_img, b_pos, closest_pos_green, (0, 255, 0), 2)
            cv2.putText(combined_img, str(key_base), b_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2 + 1)

        self.dipoles_shiffted = dipoles_shifting
        MAN.store_image(combined_img, 'shift_img')
        
    #***********************************************************************************************
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
        # crop the bump
        bump_crop = bump_img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        # crop the same location from Iref
        original_crop = iref_img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]

        bump_crop = cv2.cvtColor(bump_crop, cv2.COLOR_BGR2GRAY)
        original_crop = cv2.cvtColor(original_crop, cv2.COLOR_BGR2GRAY)

        if binary_evaluation:
            bump_crop = cv2.threshold(bump_crop, 50, 255, cv2.THRESH_BINARY_INV)[1]
            original_crop = cv2.threshold(original_crop, 50, 255, cv2.THRESH_BINARY_INV)[1]

        # if show_crops:
        #     display_images_compare(bump_crop, original_crop, f'bump_{box[2]}x{box[3]}')
        #     display_images_compare(bump_crop, original_crop)

        # 3.
        #res = cv2.norm(bump_crop, original_crop, cv2.NORM_L2)

        # 2.
        # cv2.TM_CCOEFF cv2.TM_CCOEFF_NORMED cv2.TM_CCORR cv2.TM_CCORR_NORMED cv2.TM_SQDIFF cv2.TM_SQDIFF_NORMED'
        # res = cv2.matchTemplate(bump_crop, original_crop, cv2.TM_CCORR_NORMED) 
        # print(res)
        # res = res[0]

        # 1. 
        # normalize both crops between 0 and 1
        bump_crop = bump_crop / np.linalg.norm(bump_crop) 
        original_crop = original_crop / np.linalg.norm(original_crop) 
        
        intersection = np.minimum(bump_crop, original_crop)
        #intersection = np.multiply(bump_crop, original_crop)
        union = np.maximum(bump_crop, original_crop)
        res = np.sum(intersection) / np.sum(union)

        # res = res / (box[2]*box[3])

        return res

    def evaluate_bumps(self, large_boxes, iref_img, bump_img, reverse_evaluation=False, binary_evaluation=False):
        correlations = []
        with_contours = bump_img.copy()
        i = 1
        for box in large_boxes:
            x, y, w, h = box
            corr = self.evaluate_similarity(box, bump_img, iref_img, binary_evaluation)
            correlations.append(corr)
            # draw boxes on image
            cv2.rectangle(with_contours, (x,y), (x+w,y+h), (0,255,0), 5) 
            cv2.putText(with_contours, str(i), (x+20, y+(h//2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
            if reverse_evaluation:
                volume = (corr - self.reg_b) / self.reg_m
                volume = round(volume, 2)
                str_volume = f'{volume} cm^3'
                str_corr = format(corr,'.1E') #'{:.7f}'.format(corr)
                cv2.putText(with_contours, str_corr, (x, y+h+60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3, cv2.LINE_AA)
                # cv2.putText(with_contours, str_volume, (x, y+h+60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3, cv2.LINE_AA)
            else:
                str_corr = format(corr,'.1E') #'{:.7f}'.format(corr)
                cv2.putText(with_contours, str_corr, (x, y+h+60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3, cv2.LINE_AA)
            i+=1

        print('\ncorrelation values:')
        for i in range(len(large_boxes)):
            print(f'\t{large_boxes[i][2]}x{large_boxes[i][3]}px --> {correlations[i]}')

        return correlations, with_contours

    def set_avg_correlations(self, corr_ltr, corr_rtl):
        both_directions = np.array([corr_ltr, corr_rtl])
        mean_values = np.mean(both_directions, axis=0)
        self.cal_bumps_similarity = mean_values

    def set_calibration_bumps(self, dimensions):
        h_array = []
        d_array = []
        v_array = []
        for dim in dimensions:
            h_array.append(dim[0])
            d_array.append(dim[1])
            volume = math.pi * (dim[1]/2)**2 * dim[0]
            v_array.append(volume)
        v_array = np.sort(v_array)
        
        self.cal_bumps_height = np.array(h_array)
        self.cal_bumps_diameter = np.array(d_array)
        self.cal_bumps_volume = np.array(v_array)

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