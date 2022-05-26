from unittest import result
import cv2
from cv2 import threshold
import matplotlib.pyplot as plt
import numpy as np
import os

# Loading images
lst_rois = []
refPt = []
cropping = False
def clickCrop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, lst_rois
    # if the left mouse button was clicked, record the starting (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
        cv2.rectangle(img, refPt[0], refPt[1], (55, 67, 255), 2)
        cv2.imshow("Input image", img)
        lst_rois.append(refPt)

index_img = 7 # can change
index_bg = 6 # can change

# Path of images
path_img = '.\images\img-' + str(index_img) + '.png'
path_background = '.\images\img-background-' + str(index_bg) + '.png'

# Read original image
img = cv2.imread(path_img)
originalImage, clone = img.copy(), img.copy()

# Read background Image
background = cv2.imread(path_background)
backgroundImg = background.copy()

cv2.imshow("Input image", img)
cv2.setMouseCallback("Input image", clickCrop)

# Choose some pixel in background
def getRois(lst_rois):
    while True:
        # display the image and wait for a keypress
        cv2.imshow("Input image", img)

        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
            rois = []
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
        
    # if there are two reference points, then crop the region of interest from teh image and display it
    rois = []
    for item in lst_rois:
        if len(item) == 2:
            roi = clone[item[0][1]:item[1][1], item[0][0]:item[1][0]]
            rois.append(roi.copy())
            cv2.imshow("ROI", roi)
            cv2.waitKey(0)
            
    # close all open windows
    cv2.destroyAllWindows()
    return rois
    
# Threshold calculation
def getThreshold(rois):
        
    k = 2
    obj = [[] for i in range(4)]
    
    for roi in rois:
        obj[0] += roi[:, :, 0].flatten().tolist()
        obj[1] += roi[:, :, 1].flatten().tolist()
        obj[2] += roi[:, :, 2].flatten().tolist()
    
    mean_rois = np.array([
        np.mean(np.array(obj[0])), 
        np.mean(np.array(obj[1])), 
        np.mean(np.array(obj[2]))
    ])
    
    var_rois = np.array([
        np.var(np.array(obj[0])), 
        np.var(np.array(obj[1])), 
        np.var(np.array(obj[2]))
    ])
    
    sigma = np.sqrt(var_rois)
    
    obj[3].append(mean_rois - k * sigma) # lower threshold
    obj[3].append(mean_rois + k * sigma) # upper threshold
    
    return obj[3]

# Starting with segmetation on original image
def imageSegmentation(image, lst_rois):
    img = image.copy()
    thresh = getThreshold(getRois(lst_rois))
    img[cv2.inRange(img, thresh[0], thresh[1]) != 0] = np.array([0, 0, 0])
    return img.copy()


# Background Grafting
def backgroundGrafting(oriImg, background):
    background = cv2.resize(background, (oriImg.shape[1], oriImg.shape[0]))
    for i in range(oriImg.shape[0]):
        for j in range(oriImg.shape[1]):
            if oriImg[i,j].all() == 0:
                oriImg[i,j] = background[i,j]
    return oriImg

# Image Segmentation
segImg = imageSegmentation(img, lst_rois)
segImage = segImg.copy()

# Background Grafting
bgGrafting = backgroundGrafting(segImg, background)

# Save result
os_path_result = os.listdir('.\segmentation_result')
cv2.imwrite('segmentation_result/imgResult' + str(len(os_path_result) + 1) + '.png', segImage)
os_path_bg = os.listdir(r'.\background_grafting_result')
cv2.imwrite('background_grafting_result/backgroundImgResult' + str(len(os_path_bg) + 1) + '.png', bgGrafting)

# visualizing images
def visualizing2Image(originalImg, segmentationImg, backgroundImg, backgroundGrafting):
    f = plt.figure()
    originalImg = cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB)
    segmentationImg = cv2.cvtColor(segmentationImg, cv2.COLOR_BGR2RGB)
    backgroundImg = cv2.cvtColor(backgroundImg, cv2.COLOR_BGR2RGB)
    backgroundGrafting = cv2.cvtColor(backgroundGrafting, cv2.COLOR_BGR2RGB)
    for i in range(4):
        f.add_subplot(2, 2, i + 1)
        if i == 0: 
            plt.title("Original image")
            plt.imshow(originalImg)
        elif i == 1: 
            plt.title("Image after separate background")
            plt.imshow(segmentationImg)
        elif i == 2:
            plt.title("Background image")
            plt.imshow(backgroundImg)
        else:
            plt.title("Background Grafting")
            plt.imshow(backgroundGrafting)
    plt.show(block=True)
    cv2.waitKey(0)

# Visualizing result
visualizing2Image(originalImage, segImage, backgroundImg, bgGrafting)