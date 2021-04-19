import pix2vertex as p2v
from imageio import imread
import numpy as np


from pyoints import (
    storage,
    Extent,
    transformation,
    filters,
    registration,
    normals,
    Coords,
    GeoRecords,
    nptools,
    projection,
    Grid,
    Proj,
)

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# Transforming a pix2vertex z_surface ndarray into usable dataset for ICP
def ndarray_to_coords(nd_arr):
	m,n = nd_arr.shape
	rec = nptools.recarray(
	    {'indices': np.mgrid[:m,:n].T},
	    dim=2
	)
	T = transformation.matrix(t=[1, 1], s=[1, 1])
	grid = Grid(Proj(), rec, T)
	rec = nptools.recarray({
	    'coords': np.vstack([
	            grid.records().coords[:, 0],
	            grid.records().coords[:, 1],
	            nd_arr.ravel()
	    ]).T
	})
	rec = GeoRecords(grid.proj, rec)
	return rec



front = imread("front.jpeg")
left = imread("left.jpeg")
right = imread("right.jpeg")

result_front, crop_front = p2v.reconstruct(front)
result_left, crop_left = p2v.reconstruct(left)
result_right, crop_right = p2v.reconstruct(right)



front = result_front['Z_surface']
left = result_left['Z_surface']
right = result_right['Z_surface']


# Put NAN as 0
# front = front.copy()
# front[np.isnan(front)] = front[~np.isnan(front)].min()
# left = left.copy()
# left[np.isnan(left)] = left[~np.isnan(left)].min()
# right = right.copy()
# right[np.isnan(right)] = right[~np.isnan(right)].min()


front = ndarray_to_coords(front)
left = ndarray_to_coords(left)
right = ndarray_to_coords(right)

T_front = transformation.r_matrix([90*np.pi/180, 0, 0])
T_left = transformation.r_matrix([90*np.pi/180, 0, 45*np.pi/180])
T_right = transformation.r_matrix([90*np.pi/180, 0, 315*np.pi/180])
front.transform(T_front)
left.transform(T_left)
right.transform(T_right)

colors = {'front': 'red', 'left': 'blue', 'right': 'green'}

# Ploting a rotation to on a graph
# fig = plt.figure(figsize=(15, 15))
# ax = plt.axes(projection='3d')
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_zlabel('Z (m)')
# ax.scatter(*front.coords.T, color=colors['front'])
# ax.scatter(*left.coords.T, color=colors['left'])
# ax.scatter(*right.coords.T, color=colors['right'])
# plt.show()


coords_dict = {
    'front': front.coords,
    'left': left.coords,
    'right': right.coords
}
d_th = 0.04
radii = [d_th, d_th, d_th]
icp = registration.ICP(
    radii,
    max_iter=60,
    max_change_ratio=0.000001,
    k=1
)
T_dict, pairs_dict, report = icp(coords_dict)
print(T_dict)

#Plotting a ICP graph
fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

for key in coords_dict:
    coords = transformation.transform(coords_dict[key], T_dict[key])
    ax.scatter(*coords.T, color=colors[key], label=key)
ax.legend()
plt.show()


