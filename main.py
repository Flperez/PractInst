import cv2
import numpy as np
import matplotlib.pyplot as plt


def drawPoint(image, lst_point):
    out = image.copy()
    if lst_point.size != 0:
        for xy in lst_point:
            out[xy[0],xy[1],2] = 255

    return out


def nothing(x):
    pass


if __name__=="__main__":
    cv2.namedWindow("My app")

    imgRGB = cv2.imread("images/FLIR/USER_038_VisualYCbCr888Image.bmp")

    imgtermica  = np.fromfile("images/FLIR/USER_038_ThermalRadiometricKelvin.raw",dtype=np.int16)
    imgtermica = np.reshape(imgtermica,(320,240))
    imgtermica = cv2.resize(imgtermica,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC).astype(np.float32)


    termica = np.multiply(0.01,imgtermica)

    cv2.createTrackbar("minimo", "My app", 295, 320, nothing)
    cv2.createTrackbar("maximo", "My app", 300, 320, nothing)



    while True:

        bajo = cv2.getTrackbarPos('minimo', "My app")
        alto = cv2.getTrackbarPos('maximo', "My app")

        idx_mask = np.asarray(np.where((bajo < termica) * (termica < alto))).T
        out = drawPoint(imgRGB, idx_mask)
        cv2.imshow("My app", out)
        key = cv2.waitKey(1)
        if key == ord('q')
            break









