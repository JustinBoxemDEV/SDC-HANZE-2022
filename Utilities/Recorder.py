import cv2

cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))

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