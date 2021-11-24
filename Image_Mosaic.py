# -*- coding: utf-8 -*-
"

### CV Project 2###

import numpy as np
from numpy import *
import cv2
import os
import matplotlib.pyplot as plt
import random
import math
from numpy import linalg as LA
from skimage.io import imread,imshow
from skimage import transform
import pickle
import argparse


# Harris Corner Detection
def Feature_Detection(img, k, window_size):
    row = img.shape[0]
    col = img.shape[1]
    matrix_R = np.zeros((row, col))

    cornerList = []
    # Calculate image derivatives
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate dx2, dy2 and dxy
    Ix2 = np.square(Ix)
    Iy2 = np.square(Iy)
    Ixy = Ix * Iy

    offset = int(window_size / 2)
    # Calculate Sx2, Sy2 and Sxy
    for y in range(offset, row - offset):  # rows
        for x in range(offset, col - offset):  # columns
            Sx2 = np.sum(Ix2[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sy2 = np.sum(Iy2[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sxy = np.sum(Ixy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])

            # Define matrix M at each pixel
            M = np.array([[Sx2, Sxy], [Sxy, Sy2]])

            # Calculate R score
            det = np.linalg.det(M)
            tr = np.matrix.trace(M)
            R = det - k * (tr ** 2)
            matrix_R[y, x] = R

    # print(matrix_R)
    threshold = 0.01 * matrix_R.max()
    # cv2.normalize(matrix_R, matrix_R, 0, 1, cv2.NORM_MINMAX)
    for y in range(offset, row - offset):
        for x in range(offset, col - offset):
            value = matrix_R[y, x]
            if value > threshold:
                cornerList.append([value, y, x, 1])  # 1 represents corner is detected

    return cornerList

def non_max_suppressCorners(cornerList, max_dist):
    # Sort in decreasing order
    cornerList.sort(reverse=True)

    # Mark unwanted neighbors based on Window Size
    for i in cornerList:
        index = []
        if i[3] != 0:  # Proceed only if it is corner
            for j in cornerList:
                if j[3] != 0:  # Proceed only if it is corner
                    dist = math.sqrt((j[1] - i[1]) ** 2 + (j[2] - i[2]) ** 2)
                    if (dist <= max_dist and dist > 0):
                        j[3] = 0

    # Filter out neighbors
    final = filter(lambda k: k[3] == 1, cornerList)

    return list(final)


#-------------------------------
def Feature_Matching(img1, corners1, img2, corners2, th_correlation, size):
    th = th_correlation
    match = []
    for i in range(len(corners1)):
        NCC_list = []
        x1 = corners1[i][1]
        y1 = corners1[i][2]
        NCC_max_value = 0
        isValidCorner = False
        if (x1 >= size and x1 < img1.shape[0] - size) and (y1 >= size and y1 < img1.shape[1] - size):
            for j in range(len(corners2)):
                x2 = corners2[j][1]
                y2 = corners2[j][2]
                if (x2 >= size and x2 < img2.shape[0] - size) and (y2 >= size and y2 < img2.shape[1] - size):
                    patch1 = img1[x1 - size:x1 + 1 + size, y1 - size:y1 + 1 + size]
                    patch2 = img2[x2 - size:x2 + 1 + size, y2 - size:y2 + 1 + size]


                    method = 'cv2.TM_CCORR_NORMED'
                    NCC_val = cv2.matchTemplate(patch1, patch2, eval(method))
                    # print(NCC_val, "Inbuilt")
                    NCC_list.append(NCC_val[0][0])
                    isValidCorner = True

            if isValidCorner:
                NCC_max_value = np.max(NCC_list)
                # print(NCC_max_value, "max values")
                NCC_max_index = np.argmax(NCC_list)
                # print(NCC_max_index, "max index")
                if NCC_max_value > th:
                    match.append(
                        [corners1[i][1], corners1[i][2], corners2[NCC_max_index][1], corners2[NCC_max_index][2]])
    return match

# Homography
def Apply_Ransac_new(match, N, eps):
    if len(match) > 4:

        src_pts = np.float32([[m[1],m[0]] for m in match]).reshape(-1, 1, 2)
        dst_pts = np.float32([[m[3],m[2]] for m in match]).reshape(-1, 1, 2)
        robust_H, mask_inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        mask_inliers = mask_inliers.ravel().tolist()
        matched_inliers = []
        for idx in range(len(match)):
            if mask_inliers[idx] == 1:
                matched_inliers.append(match[idx])
        return robust_H, matched_inliers

def warpImages(img2, img1, H):
    rows1, cols1 = img2.shape[:2]
    rows2, cols2 = img1.shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    # When we have established a homography we need to warp perspective
    # Change field of view
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img1, H_translation.dot(H), (x_max-x_min, y_max-y_min))

    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img2

    return output_img

def blending_Image(warped_image):
    warped_gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    warped_left_img = warped_gray[0:warped_gray.shape[0], 0:int(0.25*warped_gray.shape[1])]
    warped_left_img = warped_gray[0:warped_gray.shape[0], 0:int(0.25 * warped_gray.shape[1])]
    warped_right_img = warped_gray[0:warped_gray.shape[0], int(0.25*warped_gray.shape[1]):warped_gray.shape[1]]
    img_out_blended = cv2.addWeighted(warped_left_img, 0.7, warped_right_img, 0.3, 0)
    return img_out_blended



def plot_image(image, title, image_output_path, image_filename):
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.savefig(os.path.join(image_output_path, image_filename))

def plot_corners_image(image, title, corners, image_output_path, image_filename):
    plt.figure()
    plt.imshow(image)  # ,cmap='gray'
    x = [k[2] for k in corners]
    y = [k[1] for k in corners]
    for k in corners:
        # print(k[1])
        plt.scatter(x, y, c='r')
    plt.title(title)
    plt.savefig(os.path.join(image_output_path, image_filename))
    #plt.show()

def plot_matched_corner_image(image, title, corners, image_index, image_output_path, image_filename):
    plt.figure()
    plt.imshow(image, cmap='gray')
    if image_index == 0:
        x = [k[1] for k in corners]
        y = [k[0] for k in corners]
    else:
        x = [k[3] for k in corners]
        y = [k[2] for k in corners]
    for k in corners:
        # print(k[1])
        plt.scatter(x, y, c='r')
    plt.title(title)
    plt.savefig(os.path.join(image_output_path, image_filename))

def plot_lines_images(img1, img2, matched_corners, image_output_path, image_filename):
    plt.figure()
    plot_image = np.concatenate((img1, img2), axis=1)
    plt.imshow(plot_image)
    for k in range(len(matched_corners)):
        x_values = [matched_corners[k][1], matched_corners[k][3] + img1.shape[1]]
        y_values = [matched_corners[k][0], matched_corners[k][2]]
        plt.plot(x_values, y_values)


    #plt.plot(xvalueslist, yvalueslist)                                                                               _values_list)
    # cv2.line(plot_image,(matched_corners[k][1], ), (matched_corners[k][3]+img1.shape[1], ),(255,0,0), 1)
    plt.savefig(os.path.join(image_output_path, image_filename))


if __name__ == '__main__':
    # Parser caller arguments
    argin = argparse.ArgumentParser(description=r"""Generating Mosaic...""")
    argin.add_argument('-p', '--save_path_plot', dest='save_path_plot', type=str, required=True,
                       help='Directory to save plots')
    argin.add_argument('-t', '--dataset_type', dest='dataset_type', type=str, required=True,
                       help='Directory to save plots')

    args = argin.parse_args()
    save_path_plot = args.save_path_plot
    dataset_type = args.dataset_type
    image_path = os.path.join(save_path_plot, "Input_Images")  # Update this path if using another folder
    image_output_path = os.path.join(save_path_plot, "Output_Images")
    img_list = os.listdir(image_path)
    print(img_list)

    plt.close('all')

    if dataset_type.lower() == "hallway":
        # -----Dana Hallway Dataset -----------------
        #Left Image as Img 1
        img1_BGR = cv2.imread(os.path.join (image_path,img_list[0]))
        #Right Image as Img 2
        img2_BGR = cv2.imread(os.path.join (image_path,img_list[1]))
    elif dataset_type.lower() == "office":
        #-----Dana Office Dataset -----------------
        # Right Image as Img 1
        img1_BGR = cv2.imread(os.path.join (image_path,"DSC_0314.jpg"))
        # Left Image as Img 2
        img2_BGR = cv2.imread(os.path.join (image_path,"DSC_0313.jpg"))

    img1_RGB = cv2.cvtColor(img1_BGR, cv2.COLOR_BGR2RGB)
    img2_RGB = cv2.cvtColor(img2_BGR, cv2.COLOR_BGR2RGB)

    plot_image(img1_RGB, "Original Image 1", image_output_path, "original_img1.png")
    plot_image(img2_RGB, "Original Image 2", image_output_path, "original_img2.png")

    # RGB to Gray scale
    #img1_gray = rgb2gray(img1_rgb)
    img1_gray = cv2.cvtColor(img1_BGR, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_BGR, cv2.COLOR_BGR2GRAY)

    plt.close("all")
    img1 = img1_gray
    img2 = img2_gray

    # Obtain corner feature list
    k = 0.04
    window_size = 7
    corners1 = Feature_Detection(img1,k, window_size)
    print(len(corners1))
    corners2 = Feature_Detection(img2,k, window_size)
    print(len(corners2))

    # ---- Plotting corners with Harris Corner Detection before Non Max Suppression -----------------
    #plot_corners_image(img1_RGB, "Corners1", corners1, image_output_path, "corner1.png")
    #plot_corners_image(img2_RGB, "Corners2", corners2, image_output_path, "corner2.png")

    corner_path = os.path.join(save_path_plot, "Corners")

    max_dist = 10 #window size corresponds 7*7 for nonmax suppression
    revised_corners1 = non_max_suppressCorners(corners1, max_dist)
    revised_corners2 = non_max_suppressCorners(corners2, max_dist)

    print(len(revised_corners1))
    print(len(revised_corners2))


    #---- Plotting corners after Non Max Suppression -----------------
    plot_corners_image(img1_RGB, "Revised Corners1", revised_corners1, image_output_path, "revised_corner1.png")
    plot_corners_image(img2_RGB, "Revised Corners2", revised_corners2, image_output_path, "revised_corner2.png")

    #potential corner matches - this list contains errors (outliers)
    print(img1.shape)
    th_correlation = 0.90
    size = 3 # this is window 7*7
    matched_corners = Feature_Matching(img1,revised_corners1,img2,revised_corners2, th_correlation, size)
    print(matched_corners)
    print(len(matched_corners))

    plot_lines_images(img1_RGB, img2_RGB, matched_corners, image_output_path, "matched_corners.png")

    #---------------
    N = 1000
    eps = 320
    robust_H , matched_inliers = Apply_Ransac_new(matched_corners, N, eps )
    #robust_H , matched_inliers = Apply_Ransac(matched_corners, N, eps )
    print(matched_inliers)
    print("length of inliers", len(matched_inliers))

    plot_lines_images(img1_RGB, img2_RGB, matched_inliers, image_output_path, "matched_inliers.png")


    #----- Warping and Blending-------------------------
    warp_output_image = warpImages(img2_RGB, img1_RGB, robust_H)
    plot_image(warp_output_image, "Warped Image", image_output_path, "warped_output.png")

