import torch
import numpy
from PIL import Image
import glob
import os, os.path


# Way to retreive our data from system in some format
image_list = []
for image in glob.glob('images/*.png'): # jpg or png
    imported_image = Image.open(image).convert('RGB')
    numpy_data = numpy.asarray(imported_image)
    image_list.append(numpy_data)



def debugPrintImages():
    for image in image_list:
        print(image.shape) # (Tall, Wide, Dimensions)
        print(image)


def luminosity(rgb_arr) -> int:
    # 0.2126*R + 0.7152*G + 0.0722*B) => standard for certain colour spaces
    # 0.299*R + 0.587*G + 0.114*B = > perceived option 2, slower to calculate
    # sqrt( 0.299*R^2 + 0.587*G^2 + 0.114*B^2 ) => perceived option 2, slower to calculate
    return (0.2126 * rgb_arr[0] + 0.7152 * rgb_arr[1] + 0.7152 * rgb_arr[2])




debugPrintImages()

# Step 1: Depth determination, calculate the 3D depth from any given image
# Data structure: Voxel (volumetric pixel) array
# https://www.kaggle.com/kmader/cnn-for-generating-depth-maps-from-rgb-images

# https://towardsdatascience.com/depth-estimation-on-camera-images-using-densenets-ac454caa893
# https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e
# https://github.com/priya-dwivedi/Deep-Learning/tree/master/depth_estimation
# https://github.com/ialhashim/DenseDepth

# https://github.com/facebookresearch/meshrcnn








