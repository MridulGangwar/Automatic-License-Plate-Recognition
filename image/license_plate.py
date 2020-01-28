from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import numpy as np
import cv2
import imutils

LicensePlate = namedtuple("LicensePlateRegion",["success","plate","thresh","candidates"])

class LicensePlateDetector:
    def __init__(self,image,minPlateW=60,minPlateH=20,minChars=7,minCharW=40):
        self.image = image
        self.minPlateW = minPlateW
        self.minPlateH = minPlateH
        self.minChars = minChars
        self.minCharW = minCharW


    def detect(self):
        regions =  self.detectPlates()

        for region in regions:
            lp = self.detectCharacterCandidates(region)
            if lp.success:
                chars = self.scissor(lp)
                yield(region,chars)


    def detectPlates(self):
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(13,5))
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        regions=[]

        gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,rectKernel)

        light = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,squareKernel)
        light = cv2.threshold(light,50,255,cv2.THRESH_BINARY)[1]

        gradX = cv2.Sobel(blackhat,ddepth=cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F, dx=1, dy =0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal,maxVal) = (np.min(gradX),np.max(gradX))
        gradX = (255*((gradX-minVal)/(maxVal - minVal))).astype("uint8")

        gradX = cv2.GaussianBlur(gradX,(5,5),0)
        gradX = cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectKernel)
        thresh = cv2.threshold(gradX,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        thresh = cv2.erode(thresh,None,iterations=2)
        thresh = cv2.dilate(thresh,None,iterations=2)

        thresh = cv2.bitwise_and(thresh,thresh,mask=light)
        thresh = cv2.dilate(thresh,None,iterations=2)
        thresh = cv2.erode(thresh,None,iterations=1)

        cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            (w,h) = cv2.boundingRect(c)[2:]
            aspectRatio = w/float(h)
            
            shapeArea = cv2.contourArea(c)
            bboxArea = w*h
            extent = shapeArea/float(bboxArea)
            extent = int(extent*100)/100

            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))

            if (aspectRatio>3 and aspectRatio<6) and h>self.minPlateH and w>self.minPlateW and extent > 0.5:
                regions.append(box)

        return regions

    
    def detectCharacterCandidates(self,region):
        plate = perspective.four_point_transform(self.image,region)

        V = cv2.split(cv2.cvtColor(plate,cv2.COLOR_BGR2HSV))[2]
        T = threshold_local(V,29,offset=15,method="gaussian")
        thresh = (V>T).astype("uint8") *255
        thresh = cv2.bitwise_not(thresh)

        plate = imutils.resize(plate,width=400)
        thresh = imutils.resize(thresh,width=400)



        labels = measure.label(thresh,neighbors=8,background=0)
        charCandidates = np.zeros(thresh.shape,dtype="uint8")

        for label in np.unique(labels):
            if label ==0:
                continue
            
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            
            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
                
                aspectRatio = boxW / float(boxH)
                solidity = cv2.contourArea(c) / float(boxW * boxH)
                heightRatio = boxH / float(plate.shape[0])
                
                keepAspectRatio = aspectRatio < 1.0
                keepSolidity = solidity > 0.15
                keepHeight = heightRatio > 0.4 and heightRatio < 0.95
                
                if keepAspectRatio and keepSolidity and keepHeight:
                    hull = cv2.convexHull(c)
                    cv2.drawContours(charCandidates, [hull], -1, 255, -1)

        charCandidates = segmentation.clear_border(charCandidates)
	
        cnts = cv2.findContours(charCandidates.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if len(cnts) > self.minChars:
            (charCandidates, cnts) = self.pruneCandidates(charCandidates, cnts)
            
        thresh = cv2.bitwise_and(thresh, thresh, mask=charCandidates)
        cv2.imshow("Char Threshold", thresh)
        
        return LicensePlate(success=True,plate=plate,thresh=thresh,candidates=charCandidates)
    
    
    def pruneCandidates(self, charCandidates, cnts):
        prunedCandidates = np.zeros(charCandidates.shape, dtype="uint8")
        dims = []
        
        for c in cnts:
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            dims.append(boxY + boxH)
            
        dims = np.array(dims)
        diffs = []
        selected = []
        
        for i in range(0, len(dims)):
            diffs.append(np.absolute(dims - dims[i]).sum())
            
        for i in np.argsort(diffs)[:self.minChars]:
            cv2.drawContours(prunedCandidates, [cnts[i]], -1, 255, -1)
            selected.append(cnts[i])
            
        return (prunedCandidates, selected)
    
    
    def scissor(self, lp):
        cnts = cv2.findContours(lp.candidates.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        boxes = []
        chars = []
        
        for c in cnts:
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            dX = min(self.minCharW, self.minCharW - boxW) // 2
            boxX -= dX
            boxW += (dX * 2)
            boxes.append((boxX, boxY, boxX + boxW, boxY + boxH))
            
        boxes = sorted(boxes, key=lambda b:b[0])
        
        for (startX, startY, endX, endY) in boxes:
            chars.append(lp.thresh[startY:endY, startX:endX])
            
        return chars

    def preprocessChar(char):
        cnts = cv2.findContours(char.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts)==0:
            return None

        c = max(cnts,key=cv2.contourArea)
        (x,y,w,h) = cv2.boundingRect(c)
        char = char[y:y+h,x:x+w]
        return char



