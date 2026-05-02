import cv2
import sys
import os

input_path = "output/surveillance_video.mp4"
output_path = "output/surveillance_video.avi"

if not os.path.exists(input_path):
    print("Cannot find mp4")
    sys.exit(1)

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Cannot open original mp4")
    sys.exit(1)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    out.write(frame)

cap.release()
out.release()
print("Conversion done.")
