from os import listdir
import cv2
# from SelfDriveModel import SelfDriveModel
import torch

model_name = ""
dev = "cpu"

# model = SelfDriveModel(gpu=False)
# model.load_state_dict(torch.load(f"assets/models/{model_name}.pt", map_location=dev))
# model.eval()
# model.to(dev)

for image in listdir("D:/SDC/sdc_data/justin_data/original/images 30-03-2022 15-17-40/"):
    img = cv2.imread(f"D:/SDC/sdc_data/justin_data/original/images 30-03-2022 15-17-40/{image}")
    # print(img)

    # steer, throttle = model(img)
    steer, throttle = 0.4, 100

    cv2.putText(img, f'{steer:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.rectangle(img, (50, 400), (70, int(400-(100*throttle))), (0, 0, 255), cv2.FILLED)
    cv2.line(img, (int(848//2), int(480)), (int(848*(1+steer)//2), int(480//2)), (0, 0, 255), 2)
    cv2.imshow('Kartview', img)
    cv2.waitKey(1000//60)