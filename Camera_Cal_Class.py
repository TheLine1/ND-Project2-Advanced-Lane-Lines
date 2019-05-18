#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
class CameraCalibration():
    
    def cal_camera(self):
        # Imagenamen mit glob laden und in einer Liste abspeichern
        images = glob.glob('camera_cal/*.jpg')
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        #6*8 Kästchen -> 3 steht für x,y,z
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
        # Step through the list and search for chessboard corners
        # idx = index
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #print(idx)
    
            # Find the chessboard corners 
            # ret = bool -> gefunden oder nicht gefunden
            # corners = coordinaten der gefundenen Ecken
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (9,6), corners, ret)
                # Bilder Speichern
                write_name = 'output_images/Camera_Cal/Corners_Found/9Corner/corners_found_'+str(idx)+'.jpg'
                cv2.imwrite(write_name,img)
            
            else:
                print("Error -- Can't find Chessboard in Image" , fname)
                
        return objpoints, imgpoints
    
    def cal_result_save(self,objpoints, imgpoints):
        # Test undistortion on an image
        img = cv2.imread('camera_cal/calibration2.jpg')
        img_size = (img.shape[1], img.shape[0])
        
        # Do camera calibration given object points and image points
        # mtx -> Camera Matrix
        # dist -> Distortion Coeffition

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite('camera_cal/NEW_undist_calibration2.jpg',dst)
        
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open("Camera_Calib_Values/wide_dist_pickle9Corner_class.p", "wb" ) )
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        dist_pickle
        
        return img, dst
    
    def cal_check_all(self, objectpoints, imgpoints):
        # Check and Display Lane Images - Original and Undistorted Images
        image_orig =  glob.glob('test_images/*.jpg')
        
        for idx, fname in enumerate(image_orig):
            img_BGR = cv2.imread(fname)
            img=cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
            img_size = (img.shape[1], img.shape[0])
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
            dst = cv2.undistort(img_BGR, mtx, dist, None, mtx)
            cv2.imwrite('output_images/Camera_Cal/Undistored_Img/Undistort_Road '+ str(idx) + '.png',dst)
            dst_RGB = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
            
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
            f.tight_layout()
            ax1.imshow(img)
            ax1.set_title('Original', fontsize=15)
            ax2.imshow(dst_RGB)
            ax2.set_title('Undistort Image', fontsize=15)
        
        image_orig =  glob.glob('test_images/signs_vehicles_xygrad.png')
        
        for idx, fname in enumerate(image_orig):
            img_BGR = cv2.imread(fname)
            img=cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
            img_size = (img.shape[1], img.shape[0])
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
            dst = cv2.undistort(img_BGR, mtx, dist, None, mtx)
            cv2.imwrite('output_images/Camera_Cal/Undistored_Img/Undistort_Road_png '+ str(idx) + '.png',dst)
            dst_RGB = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
            
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
            f.tight_layout()
            ax1.imshow(img)
            ax1.set_title('Original', fontsize=15)
            ax2.imshow(dst_RGB)
            ax2.set_title('Undistort Image', fontsize=15)
        
       


# In[14]:

# =============================================================================
# 
# get_ipython().run_line_magic('matplotlib', 'inline')
# cal = CameraCalibration()
# objpoints, imgpoints = cal.cal_camera()
# img, dst = cal.cal_result_save(objpoints, imgpoints)
# #Uncoment the next row to check all road images
# #cal.cal_check_all(objpoints, imgpoints)
# 
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# ax1.imshow(img)
# ax1.set_title('Original Image', fontsize=30)
# ax2.imshow(dst)
# ax2.set_title('Undistorted Image', fontsize=30)
# 
# 
# =============================================================================
