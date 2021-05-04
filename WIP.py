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
           padding=0, min_x=None, max_x=None, min_y=None, max_y=None):
    
    # stitch in specific region specified by landmarks_to_apply
    src_x = int(average(src_landmarks[i][0] for i in landmarks_to_apply))
    src_y = int(average(src_landmarks[i][1] for i in landmarks_to_apply))
    des_x = int(average(des_landmarks[i][0] for i in landmarks_to_apply))
    des_y = int(average(des_landmarks[i][1] for i in landmarks_to_apply))
    
    src_min_x = min(src_landmarks[i][0] for i in landmarks_to_apply) - padding
    src_max_x = max(src_landmarks[i][0] for i in landmarks_to_apply) + padding
    src_min_y = min(src_landmarks[i][1] for i in landmarks_to_apply) - padding
    src_max_y = max(src_landmarks[i][1] for i in landmarks_to_apply) + padding

    des_min_x = min_x or min(des_landmarks[i][0] for i in landmarks_to_apply) - padding
    des_max_x = max_x or max(des_landmarks[i][0] for i in landmarks_to_apply) + padding
    des_min_y = min_y or min(des_landmarks[i][1] for i in landmarks_to_apply) - padding
    des_max_y = max_y or max(des_landmarks[i][1] for i in landmarks_to_apply) + padding
    
    for i,x in zip(range(des_min_x, des_max_x), range(src_min_x, src_max_x)):
        for j,y in zip(range(des_min_y, des_max_y), range(src_min_y, src_max_y)):
            des[j,i] = average( [des[j,i],src[y,x]], [0.9, 0.1] )

    return des_min_y, des_max_y


def apply_left(left, left_landmarks, des, des_landmarks):
    # apply an area around each landmark from src image to des image
    # src and des are both 2D depth arrays from pix2vert
    


    # stitch left eye
    left_eye_min_y, left_eye_max_y = stitch(left, left_landmarks, des, des_landmarks, range(36,42), padding=5)
    

    # stitch left cheek
    stitch(left, left_landmarks, des, des_landmarks, (0,1,2,3,4,5,6,7,31,40,41), min_y=left_eye_max_y)
    # left_cheek_x = int(average(left_landmarks[i][0] for i in (1, 31)))
    # left_cheek_y = int(average(left_landmarks[i][1] for i in (1, 31)))
    # des_cheek_x = int(average(des_landmarks[i][0] for i in (1, 31)))
    # des_cheek_y = int(average(des_landmarks[i][1] for i in (1, 31)))

    # cheek_min_x = min(des_landmarks[i][0] for i in (1, 31))
    # cheek_max_x = max(des_landmarks[i][0] for i in (1, 31))
    # cheek_min_y = left_eye_max_y 
    # cheek_max_y = max(des_landmarks[i][1] for i in range(5,8))

    # for i in range(-1*(cheek_max_x - cheek_min_x)//2, (cheek_max_x - cheek_min_x)//2):
    #     for j in range(-1*(cheek_max_y - cheek_min_y)//2, (cheek_max_y - cheek_min_y)//2):
    #         a,b = left_cheek_x+i, left_cheek_y+j
    #         c,d = des_cheek_x+i, des_cheek_y+j
    #         des[d,c] = average( [des[d,c],left[b,a]], [0.9, 0.1] )

    # r = 30
    # for i in range(-r, r):
    #     for j in range(-r, r):
    #         a,b = left_cheek_x+i, left_cheek_y+j
    #         c,d = des_cheek_x+i, des_cheek_y+j
    #         des[d,c] = average( [des[d,c],left[b,a]], [0.9, 0.1] )

    # stitch in left forehead
    # forehead_landmarks = (17, 18, 19, 20, 21, 68, 69, 70, 75, 76)
    # stitch(left, left_landmarks, des, des_landmarks, forehead_landmarks, max_y=left_eye_min_y)
    # left_forehead_x = int(average(left_landmarks[i][0] for i in forehead_landmarks))
    # left_forehead_y = int(average(left_landmarks[i][1] for i in forehead_landmarks))
    # des_forehead_x = int(average(des_landmarks[i][0] for i in forehead_landmarks))
    # des_forehead_y = int(average(des_landmarks[i][1] for i in forehead_landmarks))
    
    # forehead_min_x = min(des_landmarks[i][0] for i in forehead_landmarks)
    # forehead_max_x = max(des_landmarks[i][0] for i in forehead_landmarks)
    # forehead_min_y = min(des_landmarks[i][1] for i in forehead_landmarks)
    # forehead_max_y = left_eye_min_y
    
    # for i in range(-1*(forehead_max_x - forehead_min_x)//2, (forehead_max_x - forehead_min_x)//2):
    #     for j in range(-1*(forehead_max_y - forehead_min_y)//2, (forehead_max_y - forehead_min_y)//2):
    #         a,b = left_forehead_x+i, left_forehead_y+j
    #         c,d = des_forehead_x+i, des_forehead_y+j
    #         des[d,c] = average( [des[d,c],left[b,a]], [0.9, 0.1] )
    
    # r = 30
    # for i in range(-r, r):
    #     for j in range(-r, r):
    #         a,b = left_forehead_x+i, left_forehead_y+j
    #         c,d = des_forehead_x+i, des_forehead_y+j
    #         des[d,c] = average( [des[d,c],left[b,a]], [0.9, 0.1] )
            


    

def apply_right(right, right_landmarks, des, des_landmarks):
    pass
    # #stitch right name
    # left_eye_min_y, left_eye_max_y = stitch(left, left_landmarks, des, des_landmarks, range(36,42), padding=5)

    # # stitch left cheek
    # stitch(left, left_landmarks, des, des_landmarks, (1,31), max_y=left_eye_max_y)

    # # stitch in left forehead
    # forehead_landmarks = (17, 18, 19, 20, 21, 68, 69, 70, 75, 76)
    # stitch(left, left_landmarks, des, des_landmarks, forehead_landmarks, max_y=left_eye_min_y)

    
    
    

def main(front_path, left_path, right_path):
    front, front_landmarks, front_crop = helper(front_path)
    left, left_landmarks, _ = helper(left_path) # use landmarks 0-7(jaw), 17-21(brow), 36-41(eye), 68-77(left forehead)
    right, right_landmarks, _ = helper(right_path) # use landmarks 9-16(jaw), 22-26(brow), 42-47(eye)

    apply_left(left, left_landmarks, front, front_landmarks)
    # apply_right(right, right_landmarks, front, front_landmarks)

    # visualization using matplotlib
    # p2v.vis_depth_matplotlib(front_crop, front)

    p2v.save2stl(front, 'out.stl')
    
    
if __name__ == '__main__':
    main('ray-images/ray-front.jpeg', 'ray-images/ray-left.jpeg', 'ray-images/ray-right.jpeg')