from image import LicensePlateDetector
from image import BlockBinaryPixelSum
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import pickle
import random
import glob
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--samples", required=True,help="path to the training samples directory")
ap.add_argument("-m", "--min-samples", type=int, default=15,help="minimum # of samples per character")
args = vars(ap.parse_args())

blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)
 
alphabetData = []
digitsData = []
alphabetLabels = []
digitsLabels = []

for samplePath in sorted(glob.glob(args["samples"] + "/*")):
    
    sampleName = samplePath[samplePath.rfind("/") + 1:]
    imagePaths = list(paths.list_images(samplePath))
    imagePaths = random.sample(imagePaths, min(len(imagePaths), args["min_samples"]))
    
    for imagePath in imagePaths:
        char = cv2.imread(imagePath)
        char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
        char = LicensePlateDetector.preprocessChar(char)
        features = desc.describe(char)
        
        if sampleName.isdigit():
            digitsData.append(features)
            digitsLabels.append(sampleName)
        else:
            alphabetData.append(features)
            alphabetLabels.append(sampleName)

print("[INFO] fitting character model...")
charModel = LinearSVC(C=1.0, random_state=42)
charModel.fit(alphabetData, alphabetLabels)
 
print("[INFO] fitting digit model...")
digitModel = LinearSVC(C=1.0, random_state=42)
digitModel.fit(digitsData, digitsLabels)
 
print("[INFO] dumping character model...")
f = open("output/char.cpickle", "wb")
f.write(pickle.dumps(charModel))
f.close()
 
# dump the digit classifier to file
print("[INFO] dumping digit model...")
f = open("output/digit.cpickle", "wb")
f.write(pickle.dumps(digitModel))
f.close()

