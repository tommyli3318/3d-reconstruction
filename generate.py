import pix2vertex as p2v
from imageio import imread

image_path = 'tommy-images/middle.jpg'
image = imread(image_path)
result, crop = p2v.reconstruct(image)

# Interactive visualization in a notebook
p2v.vis_depth_interactive(result['Z_surface'])

# Static visualization using matplotlib
p2v.vis_depth_matplotlib(crop, result['Z_surface'])

# Export to STL
p2v.save2stl(result['Z_surface'], 'res.stl')