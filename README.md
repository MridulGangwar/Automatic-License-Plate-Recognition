# Automatic-License-Plate-Recognition

Automatic License Plate Recognition (ANPR) system includes these 4 steps:

Step 1: Photo acquisition
Step 2: Localizing the license plate
Step 3: Segmentation of characters from license plate
Step 4: Recognition of characters

After we acquire the data, we need to find the regions of an image that contain license plates. A few properties are 
used to extract the candidate license plate regions:
a) The license plate text is always darker than the license plate background
b) The license plate itself is approximately rectangle ,i.e., the license plate region is wider than it is tall. 

DetectPlates function in license_plate.py is extracting the candidate license plate regions for us. 
