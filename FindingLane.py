#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:58:10 2019

@author: linux
"""
# Functions
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import function as fnc
import cv2

class Lines():
    
    def __init__(self):
        
        # camera calibration parameters
        self.cam_mtx = None
        self.cam_dst = None
        # Extract left and right line pixel positions
        self.leftx = None
        self.lefty = None 
        self.rightx = None
        self.righty = None
        # Fit Polynominal and Calculate Curvature
        self.left_fit = None
        self.right_fit = None
        self.ploty = None
        
        self.left_curverad = None 
        self.right_curverad = None

        self.Minv = None
        self.center_dist = None

    # set camera calibration parameters
    def set_cam_calib_param(self, mtx, dst):
        self.cam_mtx = mtx
        self.cam_dst = dst
        
        
        # undistort the Image
    def undistort(self, img):
        return cv2.undistort(img, self.cam_mtx, self.cam_dst, None, self.cam_mtx)
    
    # Grayscale
    def gray(self, img):
        return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # Gray Inv
    def gray_r(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #=================================================================================
    # Read in the image and convert to grayscale
    # Canny Edge Detection
    def canny_edge(self, img, low_thresh, high_thresh, sobel_kernel, plot_flag):
 
        gray = self.gray(img)

        # Define a kernel size for Gaussian smoothing / blurring
        kernel_size = sobel_kernel # Must be an odd number (3, 5, 7...)
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

        # Define our parameters for Canny and run it
        low_threshold = low_thresh
        high_threshold = high_thresh
        canny_edge_binary = cv2.Canny(blur_gray, low_threshold, high_threshold)
        
        # Display the image
        if(plot_flag == 1):
            plt.imshow(canny_edge_binary, cmap='gray')
            #fnc.plot_triple(img,canny_edge_binary,'Original','Canny Edge Detection','gray')
    
        return canny_edge_binary

    #Sobel X and Y
    def sobelx(self, img_gray, sobel_kernel=5):
        return cv2.Sobel(img_gray, cv2.CV_64F, 1,0,ksize = sobel_kernel)
    
    def sobely(self, img_gray, sobel_kernel=5):
        return cv2.Sobel(img_gray, cv2.CV_64F, 0,1,ksize = sobel_kernel)
    
    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=5, thresh=(0, 255)):
        gray_img = self.gray(img)
        # 2) Take the derivative in x or y given orient = 'x' or 'y
        # 3) Take the absolute value of the derivative or gradient
        if orient == 'x':
            # for X direction
            sobelx = self.sobelx(gray_img,sobel_kernel)
            abs_sobel = np.absolute(sobelx)
        else:
            # for Y direction
            sobely = self.sobely(gray_img,sobel_kernel)
            abs_sobel = np.absolute(sobely)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # 5) Create a mask of 1's where the scaled gradient magnitude 
                # is > thresh_min and < thresh_max
        grad_binary_output = np.zeros_like(scaled_sobel)
        # 6) Return this mask as your binary_output image
        grad_binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return grad_binary_output
    
    def mag_thresh(self, img, sobel_kernel=5, mag_thresh=(0, 255)):
        # Calculate gradient magnitude
        # 1) Convert to grayscale
        gray_img = self.gray(img)
        # 2) Take the gradient in x and y separately
        sobelx = self.sobelx(gray_img,sobel_kernel)
        sobely = self.sobely(gray_img,sobel_kernel)
        # 3) Calculate the magnitude 
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # 5) Create a binary mask where mag thresholds are met
        scaled_sobel = np.uint8(255*gradmag/np.max(gradmag))
        # 6) Return this mask as your binary_output image
        mag_binary = np.zeros_like(gradmag)
        mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        return mag_binary

    def dir_threshold(self, img, sobel_kernel, thresh=(0, np.pi/2)):
        # Calculate gradient direction
        # 1) Convert to grayscale
        gray_img = self.gray(img)
        # 2) Take the gradient in x and y separately
        sobelx = self.sobelx(gray_img,sobel_kernel)
        sobely = self.sobely(gray_img,sobel_kernel)
        # 3) Take the absolute value of the x and y gradients
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        # 5) Create a binary mask where direction thresholds are met
        dir_binary =  np.zeros_like(absgraddir)
        dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image    
        return dir_binary

################################ Color Spaces ############################################
    def binary_thresh(self,img, thresh,plot_flag):
        gray_img = self.gray(img)
        binary = np.zeros_like(gray_img)
        binary[(gray_img > thresh[0]) & (gray_img <= thresh[1])] = 1
        
        if(plot_flag == 1):
            fnc.plot_triple(img,gray_img,binary,'Original','Gray','Binary','gray')
        
        return binary
    
    def rgb_thresh(self,img, thresh,plot_flag):
        R = img[:,:,0]
        G = img[:,:,1]
        B = img[:,:,2]

        if(plot_flag == 1):
            fnc.plot_triple(R,G,B,'R','G','B','gray')

        R_img = np.copy(R)
        R_binary = np.zeros_like(R_img)
        R_binary[(R_img> thresh[0]) & (R_img <= thresh[1])] = 1
        return R_binary
    
    def hls_thresh(self,img, thresh,plot_flag):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        H = hls[:,:,0]
        L = hls[:,:,1]
        S = hls[:,:,2]

        if(plot_flag == 1):
            fnc.plot_triple(H,L,S,'H','L','S','gray')

        hls_s_binary = np.zeros_like(S)
        hls_s_binary[(S > thresh[0]) & (S <= thresh[1])] = 1
        
        hls_h_binary = np.zeros_like(H)
        hls_h_binary[(H > thresh[2]) & (H <= thresh[3])] = 1
        return hls_s_binary, hls_h_binary 

    # pipline for binary Combination
    def pipline_combination(self, undist_Img, sobel_kernel, thresh, plot_flag):
        sobel_thresh = thresh[0]
        mag_thresh   = thresh[1]
        dir_thresh   = thresh[2]
        bin_thresh   = thresh[3]
        rgb_thresh   = thresh[4]
        hls_thresh   = thresh[5] 
        #hls_H_thresh = thresh[6]
#        gradx = line.abs_sobel_thresh(undist_Img, orient='x', sobel_kernel=ksize, thresh=(30,220))

        gradx = self.abs_sobel_thresh(undist_Img, orient='x', sobel_kernel = sobel_kernel, thresh=sobel_thresh)
        grady = self.abs_sobel_thresh(undist_Img, orient='y', sobel_kernel = sobel_kernel, thresh=sobel_thresh) 

        mag_binary = self.mag_thresh(undist_Img, sobel_kernel = sobel_kernel, mag_thresh=mag_thresh)
        dir_binary = self.dir_threshold(undist_Img, sobel_kernel = sobel_kernel, thresh=dir_thresh)

        #Binary Threshold
        binary_img = self.binary_thresh(undist_Img, bin_thresh, plot_flag)
        R_binary = self.rgb_thresh(undist_Img,rgb_thresh, plot_flag)
        hls_s_binary, hls_h_binary  = self.hls_thresh(undist_Img, hls_thresh, plot_flag)

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        combined_Color = np.zeros_like(dir_binary)
        combined_Color[((hls_s_binary == 1) & (hls_h_binary == 1)) | (R_binary == 1)] = 1
        combined_beide = np.zeros_like(dir_binary)
        combined_beide[((combined_Color == 1) | (combined == 1))] = 1

        return combined_beide

    def perspective_transform(self, img, vertices):
        # define vertices for perspective transformation
        x_size = img.shape[1]
        y_size = img.shape[0]
        img_size = (x_size, y_size)
        print(img.shape)
        #plt.imshow(img,cmap='gray')
        #plt.show()
        src = np.float32(vertices)
        #dst = np.float32([[x_size/3,y_size],[x_size/3,0],[x_size-x_size/3,0],[x_size-x_size/3, y_size]])
        dst = np.float32([[430,y_size],[430,0],[850,0],[850, y_size]])
        # Compute the perspective transform, M
        M = cv2.getPerspectiveTransform(src, dst)

        # Die Inverse der Matix M:
        # Could compute the inverse also by swapping the input parameters
        self.Minv = cv2.getPerspectiveTransform(dst, src)

        if len(img.shape) > 2:
            warped = cv2.warpPerspective(img, M, img.shape[-2:None:-1], flags=cv2.INTER_LINEAR)
        else:
            warped = cv2.warpPerspective(img, M, img.shape[-1:None:-1], flags=cv2.INTER_LINEAR)
        
        warped = cv2.warpPerspective(img,M,img_size, flags = cv2.INTER_LINEAR)

        return warped

    def hist(self, img, plot_flag):
        #print('img.shape hist:' + str(img.shape))
        # Lane lines are likely to be mostly vertical nearest to the car
        bottom_half = img[img.shape[0]//2:,:]
        #print('bottom_half'+str(bottom_half))
        # i.e. the highest areas of vertical lines should be larger values
        histogram = np.sum(bottom_half, axis=0)
        
        if(plot_flag == 1):
            plt.imshow(img, cmap='gray')
            plt.show()
            plt.plot(histogram)
            plt.show()

        return histogram
    
    def find_pixels(self, warped_bin_img, nwindows, margin, minpix, plot_flag):
        # Take a histogram of the bottom half of the image
        histogram = self.hist(warped_bin_img, plot_flag)
        
        # Create an output image to draw on and visualize the result
        # create an image to draw the lines on
        warp_zero = np.zeros_like(warped_bin_img).astype(np.uint8)
        self.out_img = np.dstack((warp_zero, warp_zero, warp_zero))

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        #nwindows = 9
        # Set the width of the windows +/- margin
        #margin = 100
        # Set minimum number of pixels found to recenter window
        #minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(warped_bin_img.shape[0]//nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped_bin_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

         # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        rectangle_data = []   



        for window in range(nwindows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped_bin_img.shape[0] - (window+1)*window_height
            win_y_high = warped_bin_img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))

            # Draw the windows on the visualization image
            cv2.rectangle(self.out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(self.out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]

            #print("goodLeft " + str(good_left_inds))
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        #images=[warped_bin_img,self.out_img]
        #titels=['warped_bin_img2','out_img2']
        #fnc.plot_n(images,titels,'gray')

        #print('warped_bin_img.shape'+str(warped_bin_img.shape))
        #print('out_img.shape'+str(self.out_img.shape))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        self.leftx = nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds] 
        self.rightx = nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds]

        return self.out_img

    def fit_polinominal(self, warped_bin_img,find_pixel_out_img, plot_flag):

        # Fit a second order polynomial to each using `np.polyfit`
        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)

        # Generate x and y values for plotting
        self.ploty = np.linspace(0, warped_bin_img.shape[0]-1, warped_bin_img.shape[0] )

        try:
            left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*self.ploty**2 + 1*self.ploty
            right_fitx = 1*self.ploty**2 + 1*self.ploty
        
        # Visualization #
        # Colors in the left and right lane regions
        self.out_img[self.lefty, self.leftx] = [255, 0, 0]
        self.out_img[self.righty, self.rightx] = [0, 0, 255]
        
        if(plot_flag == 1):
           # Create an image to draw the lines on
        #warp_zero = np.zeros_like(binary_img).astype(np.uint8)
        #scolor_warp = np.dstack((warp_zero, warp_zero, warp_zero))
            # Plots the left and right polynomials on the lane lines
            plt.plot(left_fitx, self.ploty, color='yellow')
            plt.plot(right_fitx, self.ploty, color='yellow')
            plt.imshow(self.out_img)
            plt.show()
            #print(self.out_img)

        return self.out_img

    def calc_curvature_ooc(self, bin_img):
        
        # Fit a second order polynomial to pixel positions in each fake lane line
        #left_fitx, right_fitx, ploty are from fit_ploynominal
        self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
        
        #self.left_fitx = np.polyfit(self.ploty,  self.left_fit, 2)
        #self.right_fitx = np.polyfit(self.ploty, self.right_fit, 2)
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 3.048/100#30/720 # meters per pixel in y dimension # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
        xm_per_pix = 3.7/870 #3.7/700 # meters per pixel in x dimension # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters 

        
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        
        y_eval = np.max(self.ploty)
        
        #ANDERER CODEs
        # calculate the offset from the center of the road
        y_eval = y_eval*ym_per_pix
        # calculation of R_curve (radius of curvature) 
        self.left_curverad = ((1 + (2*self.left_fit[0]*y_eval*ym_per_pix + self.left_fit[1])**2)**1.5) / np.absolute(2*self.left_fit[0])
        self.right_curverad = ((1 + (2*self.right_fit[0]*y_eval*ym_per_pix + self.right_fit[1])**2)**1.5) / np.absolute(2*self.right_fit[0])

        # Calculate vehicle center
        #left_lane and right lane bottom in pixels
        left_lane_bottom = (self.left_fit[0]*y_eval)**2 + self.left_fit[0]*y_eval + self.left_fit[2]
        right_lane_bottom = (self.right_fit[0]*y_eval)**2 + self.right_fit[0]*y_eval + self.right_fit[2]

        # Lane center as mid of left and right lane bottom                        
        lane_center = (left_lane_bottom + right_lane_bottom)/2.0
        center_image = 640
        center = (lane_center - center_image)*xm_per_pix #Convert to meters
        position = "left" if center < 0 else "right"
        self.center = "Vehicle is {:.2f}m {}".format(center, position)
        
        # Now our radius of curvature is in meters
        return center



    def draw_lane_on_image(self, original_img, binary_img):

        new_img = np.copy(original_img)
        #if l_fit is None or r_fit is None:
         #   return original_img
        
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
        cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        h,w = binary_img.shape
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (w, h)) 
        # Combine the result with the original image
        result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
        return result

















