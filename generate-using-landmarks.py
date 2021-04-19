import pix2vertex as p2v
from imageio import imread, imwrite
from landmarks import recognize_image

image_path = 'tommy-images/middle.jpg'
image = imread(image_path)
result, crop = p2v.reconstruct(image)
front_depth_arr = result['Z_surface']

cropped_path = 'cropped.jpg'
imwrite(cropped_path, crop)

landmarks = recognize_image(cropped_path, show_image=False)
print(landmarks)