import pix2vertex as p2v
from imageio import imread, imwrite
from landmarks import recognize_image
import numpy as np
import statistics


def helper(image_path):
    image = imread(image_path)
    result, crop = p2v.reconstruct(image)
    
    cropped_path = 'temp/cropped.jpg'
    imwrite(cropped_path, crop)
    
    landmarks = recognize_image(cropped_path, show_image=False)
    
    return result['Z_surface'], landmarks, crop


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


def stitch(src, src_landmarks, des, des_landmarks, landmarks_to_apply, 
           min_x, max_x, min_y=None, max_y=None, weight=0.2):
    
    # stitch in specific region specified by landmarks_to_apply
    src_x = int(average(src_landmarks[i][0] for i in landmarks_to_apply))
    src_y = int(average(src_landmarks[i][1] for i in landmarks_to_apply))
    des_x = int(average(des_landmarks[i][0] for i in landmarks_to_apply)) # this is the center x
    des_y = int(average(des_landmarks[i][1] for i in landmarks_to_apply)) # this is the center y

    min_y = min_y or min(des_landmarks[i][1] for i in landmarks_to_apply)
    max_y = max_y or max(des_landmarks[i][1] for i in landmarks_to_apply)

    for i in range(min_x - des_x, max_x - des_x):
        for j in range(min_y - des_y, max_y - des_y):
            a,b = src_x+i, src_y+j
            c,d = des_x+i, des_y+j
            des[d,c] = average( [des[d,c],src[b,a]], [1-weight, weight] )


def main(front_path, left_path, right_path):
    front, front_landmarks, front_crop = helper(front_path)
    left, left_landmarks, _ = helper(left_path) # use landmarks 0-7(jaw), 17-21(brow), 36-41(eye), 68-77(left forehead)
    right, right_landmarks, _ = helper(right_path) # use landmarks 9-16(jaw), 22-26(brow), 42-47(eye)

    # apply left side to front
    stitch(left, left_landmarks, front, front_landmarks, (75,76,68,69,70,71, 3,4,5,6,7,8),
           min_x=front_landmarks[2][0], max_x=front_landmarks[27][0])
    
    # apply right side to front
    stitch(right, right_landmarks, front, front_landmarks, (80,72,73,79,74,71, 13,12,11,10,9,8),
           min_x=front_landmarks[27][0], max_x=front_landmarks[14][0])

    # visualization using matplotlib
    p2v.vis_depth_matplotlib(front_crop, front, stride=1)

    # save stl
    p2v.save2stl(front, 'out.stl')
    
    
if __name__ == '__main__':
    main('ray-images/ray-front.jpeg', 'ray-images/ray-left.jpeg', 'ray-images/ray-right.jpeg')