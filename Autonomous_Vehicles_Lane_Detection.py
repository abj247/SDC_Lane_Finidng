#!/usr/bin/env python
# coding: utf-8

# In[11]:


#Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, glob
from keras_preprocessing.image import load_img
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[267]:




def view_pic(imgs, cmap = None):
    columns = 2
    rows = (len(imgs)+1)//columns
    
    plt.figure(figsize = (12, 12))
    for i, img in enumerate(imgs):
        plt.subplot(rows, columns, i+1)
        if len(img.shape) == 2:
            cmap = 'gray'
        else:
            cmap
        
        plt.imshow(img, cmap = cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()
    
    
#viewing images
train_images = [plt.imread(path) for path in glob.glob(r'E:\SDC_LD\train_images\*.jpg')]

view_pic(train_images)


# In[273]:


def choose(img):
    low_lim_w = np.uint8([200, 200, 200])
    up_lim_w = np.uint8([255, 255, 255])
    show_white_lines = cv2.inRange(img, low_lim_w, up_lim_w)
    
    low_lim_y = np.uint8([170, 170, 0])
    up_lim_y = np.uint8([255, 255, 255])
    show_yellow_lines = cv2.inRange(img, low_lim_y, up_lim_y)
    
    mask = cv2.bitwise_or(show_white_lines,show_yellow_lines )
    do_masking = cv2.bitwise_and(img, img, mask = mask)
    return do_masking

view_pic(list(map(choose,train_images)))


# In[274]:


def apply_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

view_pic(list(map(apply_hsv, train_images)))


# In[275]:


def apply_hsl(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

view_pic(list(map(apply_hsl, train_images)))


# In[288]:


def choose_hsv(img):
    convert = apply_hsv(img)
    
    low_lim = np.uint8([0, 200, 0])
    up_lim = np.uint8([255, 255, 255])
    show_white_lines = cv2.inRange(img, low_lim, up_lim)
    
    low_lim = np.uint8([10, 0, 100])
    up_lim = np.uint8([40, 255, 255])
    show_yellow_lines = cv2.inRange(img, low_lim, up_lim)
    
    mask = cv2.bitwise_or(show_white_lines,show_yellow_lines )
    do_masking = cv2.bitwise_and(img, img, mask = mask)
    return do_masking

view_pic(list(map(choose_hsv, train_images)))


# In[549]:


def choose_hsl(img):
    convert = apply_hsl(img)
    
    low_lim = np.uint8([0, 200, 0])
    up_lim = np.uint8([255, 255, 255])
    show_white_lines = cv2.inRange(img, low_lim, up_lim)
    
    low_lim = np.uint8([25, 0, 100])
    up_lim = np.uint8([55, 255, 255])
    show_yellow_lines = cv2.inRange(img, low_lim, up_lim)
    
    mask = cv2.bitwise_or(show_white_lines,show_yellow_lines )
    do_masking = cv2.bitwise_and(img, img, mask = mask)
    return do_masking
new_images = list(map(choose_hsl, train_images))
view_pic(list(map(choose_hsl, train_images)))


# In[290]:


def apply_gray_scaling(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


gray_scale = list(map(apply_gray_scaling, new_images))
view_pic(gray_scale)


# In[397]:


def do_gaussian_blurring(img):
    return cv2.GaussianBlur(img, (19, 19), 0 )

gb_images = list(map(do_gaussian_blurring, gray_scale))

view_pic(gb_images)
    


# In[398]:


#Canny Edge detection
def Canny_ed(img):
    return cv2.Canny(img, 50, 150)

Canny_images = list(map(Canny_ed, gb_images))
view_pic(Canny_images)


# In[399]:


def roi_selection(img, points):
    mask = np.zeros_like(img)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, points, 255)
    else:
        cv2.fillPoly(mask, points, (255,)*mask.shape[2])
    return cv2.bitwise_and(img, mask)
                      
def region(img):
    rows, cols = img.shape[:2]
    bottom_left  = [cols*0.1, rows*0.95]
    top_left     = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right    = [cols*0.6, rows*0.6] 
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    points = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return roi_selection(img, points)
                      
                      
roi_images = list(map(region, Canny_images))

view_pic(roi_images)
                      


# In[521]:


#Probablistic_hough_line_transformation
def PHLT(img):
    return cv2.HoughLinesP(img, 1, np.pi/180, threshold = 51, minLineLength=20, maxLineGap=300)

l_lines = list(map(PHLT, roi_images))



# In[522]:


#Drawing Lines
def line_maker(img, lines,  make_copy = True):
    if make_copy:
        img = np.copy(img)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1,y1), (x2,y2), [255, 0, 0], 2)
    return img

images_of_lines = []
for img,lines in zip(train_images, l_lines):
    images_of_lines.append(line_maker(img,lines))

view_pic(images_of_lines)


# In[523]:


def find_m_and_c(lines):
    l_param = []
    l_length = []
    r_param = []
    r_length = []
    
    for line in lines:
        for x0,y0,x1,y1 in line:
            if x1 == x0:
                continue
            m = (y1-y0)/(x1-x0)
            c = y0 - m*x0
            length = np.sqrt((y1-y0)**2+(x1-x0)**2)
            if m < 0:
                l_param.append((m, c))
                l_length.append((length))
            else:
                r_param.append((m,c))
                r_length.append((length))
    left_lane  = np.dot(l_length,  l_param) /np.sum(l_length)  if len(l_length) >0 else None
    right_lane = np.dot(r_length, r_param)/np.sum(r_length) if len(r_length)>0 else None
    
    return left_lane, right_lane
    


# In[524]:


def line_to_pixel(y0, y1, line):
    
    if line is None:
        return None
    m, c = line
    
    x0 = int((y0 - c)/m)
    x1 = int((y1 - c)/m)
    y0 = int(y0)
    y1 = int(y1)
             
    return ((x0,y0), (x1,y1))


# In[537]:


def lane_lines(img, lines):
    left_lane, right_lane = find_m_and_c(lines)
    
    y0 = img.shape[0]
    y1 = y0*0.63
    
    left_side = line_to_pixel(y0,y1,left_lane)
    right_side = line_to_pixel(y0,y1, right_lane)
    
    return left_side, right_side


# In[541]:


def draw_images(img, lines, color=[20,190,255], thickness = 20):
    line_pic =np.zeros_like(img)
    for line in lines:
        if line is not None:
            cv2.line(line_pic,*line, color, thickness)
    return cv2.addWeighted(img, 0.54, line_pic, 1.0, 0.0)

images_ld = []
for img, lines in zip(train_images, l_lines):
    images_ld.append(draw_images(img, lane_lines(img, lines)))
    
view_pic(images_ld)


# In[ ]:




