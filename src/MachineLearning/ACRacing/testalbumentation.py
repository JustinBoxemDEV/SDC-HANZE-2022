import albumentations as A
import cv2

transform = A.Compose([
    # A.RandomSunFlare(flare_roi=(0, 0, 1, 0.25), angle_lower=0, angle_upper=1, num_flare_circles_lower=19, num_flare_circles_upper=20, src_radius=500),
    A.HorizontalFlip(p=1),
    # A.MotionBlur(blur_limit=10, p=1),
    # A.RandomShadow(shadow_roi=(0, 0.55, 1, 1), num_shadows_lower=0, num_shadows_upper=3, p=0.6, shadow_dimension=85),
    # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.4),
    # A.RandomGamma(gamma_limit=(120, 120), p=0.4)
])

image = cv2.imread("D:\\Github Desktop Clones\\SDC-HANZE-2022\\src\\MachineLearning\\ACRacing\\TestImges\\1.png")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

for i in range(10):

    transformed = transform(image=image)
    transformed_imaged = transformed["image"]

    cv2.imshow("Test"+str(i), transformed_imaged)
    cv2.waitKey(0)

cv2.destroyAllWindows()