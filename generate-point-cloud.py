# Generate a bunch of XYZ points based on the three input images
# Start with the dlib output and use depth information

import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib

# Point cloud
point_cloud = [] # Array of (x,y,z) tuples

def getDepth(BGR):
    B, G, R = BGR
    return round(R * 299/1000 + G * 587/1000 + B * 114/1000, 2) # grayscale intensity value of pixel

def getRadiusDepthAverage(img, x : int ,y : int , r : int) -> float:

    total_depth = 0
    num_pixels = 0

    a = np.array((x,y))
    
    for i in range(x - r, x + r):
        for j in range(y - r, y + r):
            
            if(i < 0 or i >= img.shape[1] or j < 0 or j >= img.shape[0]):
                continue
            
            b = np.array((i,j))

            if(abs(np.linalg.norm(a-b)) > r):
                continue
            
            total_depth += getDepth(img[x,y])
            num_pixels += 1
    
    return total_depth / num_pixels

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# read the image
img = cv2.imread("images/girl.jpg") # images/girl.jpg

# resize the image
# img = cv2.resize(img, (2448//4,3264//4)) # original is (2448 x 3264)

# Convert image into grayscale
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    

# Use detector to find landmarks
faces = detector(gray)
for face in faces:
    x1 = face.left() # left point
    y1 = face.top() # top point
    x2 = face.right() # right point
    y2 = face.bottom() # bottom point

    # Create landmark object
    landmarks = predictor(image=gray, box=face)

    # this dictionary maps each region to its depth estimate
    region_depths = {
        range( 0, 17) : 0.0, # jawline
        range(17, 22) : 0.0, # left eyebrow
        range(22, 27) : 0.0, # right eyebrow
        range(27, 31) : 0.0, # nose line
        range(31, 36) : 0.0, # nose
        range(36, 42) : 0.0, # left eye
        range(42, 48) : 0.0, # right eye
        range(48, 68) : 0.0  # mouth
    }

    # Loop through all the points
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        radius_z = getRadiusDepthAverage(img, x, y, 1)
        # radius_z = getDepth(img[x,y])
        for region in region_depths.keys():
            if (n in region): # if current point belongs to this region
                region_depths[region] += radius_z # add it to the total
        
        point_cloud.append( [x,y,-1] ) # -1 is placeholder

        # Draw a circle
        cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
        
        # put text
        cv2.putText(img=img, text=str(n), org=(x,y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255))
    
    # divide each total by number of points
    for region in region_depths.keys():
        region_depths[region] = region_depths[region] / len(region)
        
    # set the depth for each point in point_cloud
    for region in region_depths.keys():
        for i in region:
            point_cloud[i][2] = region_depths[region]

# show the image
# cv2.imshow(winname="Face", mat=img)

# Delay between every fram
cv2.waitKey(delay=0)

# Close all windows
cv2.destroyAllWindows()

point_cloud = np.array(point_cloud)
# print(point_cloud)
# print(point_cloud.shape)


def plotXyz(point_cloud) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = []
    Y = []
    Z = []

    # extract xyz values from self.xyz
    for xyz in point_cloud:
        x,y,z = xyz
        X.append(x)
        Y.append(y)
        Z.append(z)

    ax.scatter(X,Y,Z)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_zlim3d(-10,20)
    plt.show()


plotXyz(point_cloud)