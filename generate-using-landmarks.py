import pix2vertex as p2v
from imageio import imread, imwrite
from landmarks import recognize_image
import numpy as np


def helper(image_path):
    image = imread(image_path)
    result, crop = p2v.reconstruct(image)
    
    cropped_path = 'temp/cropped.jpg'
    imwrite(cropped_path, crop)
    
    landmarks = recognize_image(cropped_path, show_image=False)
    
    return result['Z_surface'], landmarks


def apply(src, src_landmarks, des, des_landmarks, landmarks_to_apply, r):
    # apply an area around each landmark from src image to des image

    for i in landmarks_to_apply:
        src_y, src_x = src_landmarks[i]
        des_y, des_x = des_landmarks[i]
        
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
                
                des[c,d] = src[a,b]


def main(front_path, left_path, right_path):
    front, front_landmarks = helper(front_path)
    left, left_landmarks = helper(left_path) # use landmarks 0-7(jaw), 17-21(brow), 36-41(eye)
    right, right_landmarks = helper(right_path) # use landmarks 9-16(jaw), 22-26(brow), 42-47(eye)

    apply(left, left_landmarks, front, front_landmarks, list(range(0,8))+list(range(17,22))+list(range(36,42)), 10)
    apply(right, right_landmarks, front, front_landmarks, list(range(9,17))+list(range(22,27))+list(range(42,48)), 10)

    p2v.save2stl(front, 'ray.stl')
    
    
if __name__ == '__main__':
    # main('tommy-images/middle.jpg', 'tommy-images/left.jpg', 'tommy-images/right.jpg') # errors
    main('ray-images/ray-front.jpeg', 'ray-images/ray-left.jpeg', 'ray-images/ray-right.jpeg')