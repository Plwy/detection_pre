import cv2
import numpy as np


def zooming(img, scale=0.1):
    img = cv2.resize(img,(int(scale*img.shape[1]),int(scale*img.shape[0])))
    return img


alpha = 0.3
beta = 80
# img_path = "7MeansDenoising/1_1.bmp"
img_path = 'test/zsl_test_image/template/1.jpg'
img = cv2.imread(img_path)
# img2 = cv2.imread(img_path)
img2 = img.copy()
img2 = zooming(img2)

def updateAlpha(x):
    global alpha, img, img2
    alpha = cv2.getTrackbarPos('Alpha', 'image')
    alpha = alpha * 0.01
    img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))
def updateBeta(x):
    global beta, img, img2
    beta = cv2.getTrackbarPos('Beta', 'image')
    img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))
# 创建窗口
cv2.namedWindow('image')
cv2.createTrackbar('Alpha', 'image', 0, 300, updateAlpha)
cv2.createTrackbar('Beta', 'image', 0, 255, updateBeta)
cv2.setTrackbarPos('Alpha', 'image', 100)
cv2.setTrackbarPos('Beta', 'image', 10)
while (True):
    cv2.imshow('image', img)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
