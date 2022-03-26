# -*- coding: utf-8 -*-
"""
Image Face Dectection Code

Dec 29, 2021
"""
import cv2
import face_recognition

# Loading the image to detetct
image_to_detect = cv2.imread('/Users/danielcoelho/Desktop/FaceRecognition/images/wakanda.png')

# Detect all faces in the image -- model = 'cnn' can be used but takes long, but more accurate
all_face_location = face_recognition.face_locations(image_to_detect, model = 'hog')

# Printing the number of faces detected
print('There are {} number of faces in this image'.format(len(all_face_location)))

# Loopiong through to find face location
for index, current_face_location in enumerate(all_face_location):
    
    # Spliting the tuple for each position values
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print('Found face {} at top: {}, right: {}, bottom: {}, left: {}'.format(index +1,top_pos, right_pos, bottom_pos, left_pos))
    current_face_image = image_to_detect[top_pos:bottom_pos, left_pos:right_pos]
    cv2.imshow("Face Number: " + str(index +1), current_face_image)
    
    
