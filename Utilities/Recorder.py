import cv2
import os
import sys
from datetime import datetime

currentPath = os.path.abspath(os.getcwd())

fps=60
if len(sys.argv) > 1:
    fps=sys.argv[1]
print("Recording with " + str(fps) + " fps")

dt = datetime.now()
ts = datetime.timestamp(dt)
str_date_time = dt.strftime("%d-%m-%Y-%H:%M")

cap = cv2.VideoCapture(0)
capFocus = 25  # min: 0, max: 255, increment:5
cap.set(28, capFocus) 

out = cv2.VideoWriter(currentPath + "/assets/videos/" + str_date_time +  "---" + str(fps) + "fps.mp4",cv2.VideoWriter_fourcc(*'mp4v'),int(fps), (1280, 720))

if (cap.isOpened()== False):
    print("Error opening video stream or file")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        height, width = frame.shape[:2]
        center = (width/2, height/2)
        frame = cv2.resize(frame, (1280,720))
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()