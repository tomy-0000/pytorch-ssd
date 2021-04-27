from .vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
import cv2
import sys

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = create_mobilenetv2_ssd_lite(2, is_test=True)
net.load("path to pth")
net.to(DEVICE)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=DEVICE)

def ssd(orig_image):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    return boxes.numpy()