# import the necessary packages
import numpy as np
import cv2

class BlockBinaryPixelSum:
    def __init__(self, targetSize=(30, 15), blockSizes=((5, 5),)):
        self.targetSize = targetSize
        self.blockSizes = blockSizes
        
    def describe(self, image):
        image = cv2.resize(image, (self.targetSize[1], self.targetSize[0]))
        features = []
        
        for (blockW, blockH) in self.blockSizes:
            for y in range(0, image.shape[0], blockH):
                for x in range(0, image.shape[1], blockW):
                    roi = image[y:y + blockH, x:x + blockW]
                    total = cv2.countNonZero(roi) / float(roi.shape[0] * roi.shape[1])
                    
                    features.append(total)
                                        
        return np.array(features)
