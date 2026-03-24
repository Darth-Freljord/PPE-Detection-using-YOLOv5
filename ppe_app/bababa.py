import cv2
cap = cv2.VideoCapture(0)
print("Camera still in use:", cv2.VideoCapture(0).isOpened())