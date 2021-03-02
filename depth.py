# Step 1: Depth determination, calculate the 3D depth from any given image

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
import glob


class Image:
    def __init__(self, RGB=None, intensity=None, xyz=None):
        self.RGB = RGB
        self.intensity = intensity
        self.xyz = xyz

        if self.RGB is not None:
            self.RGB = np.asarray(self.RGB)
            if self.intensity is None:
                self.intensity = self.generateIntensityArrFromRGBArr()
            
            if self.xyz is None:
                self.xyz = self.generateXyzFromIntensity()


    def generateIntensityArrFromRGBArr(self) -> np.array:
        intensity = []

        for row in self.RGB:
            temp = []
            for rgb in row: # rgb is an arr len 3 of RGB values
                temp.append(self.generateIntensityVal(rgb))
            intensity.append(temp)
        return np.asarray(intensity)


    def generateIntensityVal(self, rgb) -> float:
        # Get intensity from arr of RGB values

        # return (0.2126 * R + 0.7152 * G + 0.7152 * B) # standard for certain colour spaces

        R, G, B = rgb
        return R * 299/1000 + G * 587/1000 + B * 114/1000
        
        # return (0.299*R ** 2 + 0.587*G ** 2 + 0.114*B ** 2) ** 0.5

    def generateXyzFromIntensity(self) -> np.array:
        xyz = []
        for y, intensity_row in enumerate(self.intensity, 0):
            temp = []
            for x, intensity_val in enumerate(intensity_row):
                temp.append([int(x), int(y), round(intensity_val,2)]) # intensity_val is the z
            xyz.append(temp)
        return np.asarray(xyz)
    
    def normalize(self, arr, a, b) -> None:
        # normalizes arr to be in range of (a, b)
        A = min(arr) # old min
        B = max(arr) # old max
        
        for i in range(len(arr)):
            arr[i] = round(a + (arr[i]-A)*(b-a) / (B-A), 2)

    def plotXyz(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        X = []
        Y = []
        Z = []

        # extract xyz values from self.xyz
        for row in self.xyz:
            for x,y,z in row: # rgb is an arr len 3 of RGB values
                X.append(x)
                Y.append(y)
                Z.append(z)

        self.normalize(Z, 0, 10)

        ax.plot(X,Y,Z)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_zlim3d(-10,20)
        plt.show()

    

def main():
    # Retreive our data from system in some format
    images = []
    for image in glob.glob('images/*'): # jpg or png
        RGB_arr = PIL_Image.open(image).convert('RGB')
        images.append(Image(RGB_arr))

    for image in images:
        print("Image.RGB shape:")
        print(image.RGB.shape) # (numRows, numCols, length of arr[i][j])

        # print("Image.intensity shape:")
        # print(image.intensity.shape)
        # print("Image.intensity values:")
        # print(image.intensity)

        # print("Image.xyz shape:")
        # print(image.xyz.shape)
        # print("Image.xyz values:")
        # print(image.xyz)
        image.plotXyz()



main()
