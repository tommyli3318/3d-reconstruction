import torch
import numpy as np
from PIL import Image
import glob
import os, os.path
from DepthVisualizer.DepthVisualizer import (
    Utils as DVUtils,
    DepthRenderer as DVRenderer
) # https://github.com/ErenBalatkan/DepthVisualizer/blob/master/DepthVisualizer/DepthVisualizer.py


# Retreive our data from system in some format
image_list = []
for image in glob.glob('images/*.png'): # jpg or png
    imported_image = Image.open(image).convert('RGB')
    numpy_data = np.asarray(imported_image)
    image_list.append(numpy_data)


# Convert rgb_arr to luminosity
def luminosity(rgb_arr) -> float:
    # 0.2126*R + 0.7152*G + 0.0722*B) => standard for certain colour spaces
    # 0.299*R + 0.587*G + 0.114*B = > perceived option 2, slower to calculate
    # sqrt( 0.299*R^2 + 0.587*G^2 + 0.114*B^2 ) => perceived option 2, slower to calculate
    return (0.2126 * rgb_arr[0] + 0.7152 * rgb_arr[1] + 0.7152 * rgb_arr[2])

# Converts RGB array to 3D Point array
def convertRGBArrayTo3DPointArray(RGB_arr):
    point_array = []

    principle_x = (len(RGB_arr)) // 2
    principle_y = (len(RGB_arr[0])) // 2
    
    for i in range(RGB_arr.shape[0]):
        temp = []
        for j in range(RGB_arr.shape[1]):
            RGB = RGB_arr[i][j]
            conversion = DVUtils.convert_depth_pixel_to_point(
                x = j,
                y = i,
                depth = luminosity(RGB),
                focal_length_in_pixels = 1000, # filler
                principal_point = [principle_x, principle_y],
                rgb = RGB
            )
            temp.append(conversion)
        point_array.append(temp)
    return np.array(point_array)

# Converts 3D Point array to Voxel array
def convert3DPointersArrayToVoxelArray(points_arr):
    center_x = len(points_arr) // 2
    center_y = len(points_arr[0]) // 2
    max_x = float('-inf')
    max_y = float('-inf')
    max_z = float('-inf') 

    
    
    points_array_test = []
    for i in range(points_arr.shape[0]):
        for j in range(points_arr.shape[1]):
            points_array_test.append(points_arr[i][j])
            max_x = max(max_x, points_arr[i][j][0])
            max_y = max(max_y, points_arr[i][j][1])
            max_z = max(max_z, points_arr[i][j][2])

    center_z = max_z // 2


    # volume = max_x * max_y * max_z
    # volume/1000000

    final_voxel_array = DVUtils.convert_points_to_voxel_map(
        points = points_array_test,
        voxel_map_center = [center_x, center_y, center_z],
        voxel_map_size = [max_x, max_y, max_z],
        voxel_size = 1,
    )

    return final_voxel_array


for image in image_list:
    # print(image.shape) # (numRows, numCols, Dimensions)
    point_array = convertRGBArrayTo3DPointArray(image)
    # print(point_array)
    voxel_array = convert3DPointersArrayToVoxelArray(point_array)
    print(voxel_array)

    # renderer = DVRenderer(500,500)
    # renderer.add_voxels(voxel_array)
    # renderer.render()
    # renderer.show_window()



# Step 1: Depth determination, calculate the 3D depth from any given image
# Data structure: Voxel (volumetric pixel) array
# https://www.kaggle.com/kmader/cnn-for-generating-depth-maps-from-rgb-images
# https://towardsdatascience.com/depth-estimation-on-camera-images-using-densenets-ac454caa893
# https://github.com/ialhashim/DenseDepth
# https://github.com/facebookresearch/meshrcnn

