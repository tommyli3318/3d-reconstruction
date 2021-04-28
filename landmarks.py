import cv2
import numpy as np
import dlib

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor from https://github.com/codeniko/shape_predictor_81_face_landmarks
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

def recognize_image(image_path, show_image=False):
    # read the image and convert image into grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    res = []

    # Use detector to find landmarks
    faces = detector(gray)
    
    assert len(faces) != 0, "Error: no face detected in input image %s" % image_path
    assert len(faces) == 1, "Error: detected more than 1 face in input image %s" % image_path
    
    face = faces[0]
    x1 = face.left() # left point
    y1 = face.top() # top point
    x2 = face.right() # right point
    y2 = face.bottom() # bottom point

    # Create landmark object
    landmarks = predictor(image=gray, box=face)

    # Loop through all the points
    for n in range(0, 81):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        res.append( (x,y) )
        
        # Draw a circle
        cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
        
        # put text
        cv2.putText(img=img, text=str(n), org=(x,y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255))

    if show_image:
        # show the image
        cv2.imshow(winname="Face", mat=img)

        # Delay between every fram
        cv2.waitKey(delay=0)

        # Close all windows
        cv2.destroyAllWindows()
    
    
    return res # list of (x,y) tuples representing each landmark
