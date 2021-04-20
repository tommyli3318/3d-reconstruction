import pix2vertex as p2v
from imageio import imread, imwrite
from landmarks import recognize_image

def helper(image_path):
    image = imread(image_path)
    result, crop = p2v.reconstruct(image)
    
    cropped_path = 'temp/cropped.jpg'
    imwrite(cropped_path, crop)
    
    landmarks = recognize_image(cropped_path, show_image=True)
    
    return result['Z_surface'], landmarks

def main(front, left, right):
    front_arr, front_landmarks = helper(front)
    left_arr, left_landmarks = helper(left)
    right_arr, right_landmarks = helper(right)
    
    # print(front)
    # print(left)
    # print(right)

# main('tommy-images/middle.jpg', 'tommy-images/left.jpg', 'tommy-images/right.jpg')
main('ray-images/ray-front.jpeg', 'ray-images/ray-left.jpeg', 'ray-images/ray-right.jpeg')