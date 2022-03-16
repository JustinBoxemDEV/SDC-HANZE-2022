import cv2

cap = cv2.VideoCapture('../assets/videos/testdag160FPS.mp4')
out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 60, (1280, 720))

if (cap.isOpened()== False):
    print("Error opening video stream or file")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        height, width = frame.shape[:2]
        center = (width/2, height/2)

        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-4.8, scale=1)

        rotated_image = cv2.warpAffine(src=frame, M=rotate_matrix, dsize=(width, height))
        rotated_image = cv2.resize(rotated_image, (1280,720))
        out.write(rotated_image)

        cv2.imshow('Original image', frame)
        cv2.imshow('Rotated image', rotated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()