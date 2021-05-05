import pix2vertex as p2v
from imageio import imread, imwrite
from landmarks import recognize_image
import numpy as np
import statistics
import math


def helper(image_path):
    image = imread(image_path)
    result, crop = p2v.reconstruct(image)
    
    cropped_path = 'temp/cropped.jpg'
    imwrite(cropped_path, crop)
    
    landmarks = recognize_image(cropped_path, show_image=False)
    
    return result['Z_surface'], landmarks, crop


def apply(src, src_landmarks, des, des_landmarks, landmarks_to_apply, r):
    # apply an area around each landmark from src image to des image
    # src and des are both 2D depth arrays from pix2vert
    for i in landmarks_to_apply:
        src_x, src_y = src_landmarks[i]
        des_x, des_y = des_landmarks[i]

        # grab circular area
        for i in range(-r, r):
            for j in range(-r, r):
                a,b = src_x+i, src_y+j
                c,d = des_x+i, des_y+j
                
                if not (0 <= a < src.shape[0]) or not (0 <= b < src.shape[1]):
                    continue
                if not (0 <= c < des.shape[0]) or not (0 <= d < des.shape[1]):
                    continue
                if abs(np.linalg.norm([i,j])) > r:
                    continue
                
                des[d,c] = statistics.mean( (des[d,c], src[b,a]) )

def average(iterable, weights=None):
    if not weights:
        return statistics.mean(iterable)

    assert len(iterable) == len(weights), "Iterable and weights must have same length"
    
    for i in range(len(iterable)):
        if iterable[i] is np.NaN:
            iterable[i] = 0

    iterable = np.array(iterable)
    weights = np.array(weights)
    
    return sum(iterable * weights) / sum(weights)


def dist(p1, p2):
    return math.sqrt(abs(p1[0] - p2[0]) ** 2 + abs(p1[1] - p2[1]) ** 2)


def coreresponding(src_landmarks, des_landmarks, i, j):
    # find the a,b which would corresponding to des[i,j]
    
    # Traverse through des_landmarks and find the closest pair (x,y) to (i,j)
    # Calculate the difference from this (x,y) <- index 2 to (i,j)
    # if (x,y) = (1,2) and (i,j) = (3,4), the difference is (+1,+2)
    # src_landmarks[2] <- c+1,d+2
    
    # x,y = j,i
    # i == y
    # j == x

    index = -1
    minimum_distance = float('inf')
    
    diff = (0,0) # (x,y)
    for curr_index, (x, y) in enumerate(des_landmarks):
        new_distance = dist([x,y], [j,i])
        if new_distance < minimum_distance:
            minimum_distance = new_distance
            diff = (j-x, i-y)
            index = curr_index

    a,b = src_landmarks[index]
    return (a + diff[0], b + diff[1])

    
def apply_left(left, left_landmarks, des, des_landmarks):
    # TODO
    for i in range(des.shape[0]):
        for j in range(des.shape[1]):
            if des[i,j] == np.NaN:
                continue

            # grab the corresponding left[x,y] pixel and apply it onto des[i,j]
            x,y = coreresponding(left_landmarks, des_landmarks, i, j)
            
            if not (0 <= y < left.shape[0] and 0 <= x < left.shape[1]):
                continue
            
            # print(j,i, "corresponds to", x,y)
            if left[y,x] == np.NaN:
                continue
            
            # des[i,j] = left[y,x]
            des[i,j] = average( (des[i,j], left[y,x]), [0.9, 0.1] )




def main(front_path, left_path, right_path):
    front, front_landmarks, front_crop = helper(front_path)
    
    left, left_landmarks, _ = helper(left_path) # use landmarks 0-7(jaw), 17-21(brow), 36-41(eye)
    right, right_landmarks, _ = helper(right_path) # use landmarks 9-16(jaw), 22-26(brow), 42-47(eye)

    # apply(left, left_landmarks, front, front_landmarks, list(range(0,8))+list(range(17,22))+list(range(36,42)), 10)
    # apply(right, right_landmarks, front, front_landmarks, list(range(9,17))+list(range(22,27))+list(range(42,48)), 10)
    apply_left(left, left_landmarks, front, front_landmarks)

    # visualization using matplotlib
    # p2v.vis_depth_matplotlib(front_crop, front)

    p2v.save2stl(front, 'test.stl')
    
    
if __name__ == '__main__':
    # main('tommy-images/middle.jpg', 'tommy-images/left.jpg', 'tommy-images/right.jpg') # errors
    main('ray-images/ray-front.jpeg', 'ray-images/ray-left.jpeg', 'ray-images/ray-right.jpeg')