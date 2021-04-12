import pix2vertex as p2v
from imageio import imread

image_path = 'tommy-images/middle.jpg'
image = imread(image_path)
result, crop = p2v.reconstruct(image)
# result = {'Z': Z, 'X': X, 'Y': Y, 'Z_surface': Z_surface}

A = result['Z_surface']

for i in range(50):
    for j in range(50):
        A[i][j] = (i*j) ** 0.5

with open('out.txt', 'w') as f:
    for inner in A:
        for val in inner:
            f.write(str(val) + ' ')
        f.write('\n')


p2v.save2stl(A, 'res.stl')