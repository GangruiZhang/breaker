from tkinter import *
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np

def selectPath():
    # path_ = askdirectory()
    path_ = askopenfilename()
    path.set(path_)
    print(path_)

    def stackImages(scale, imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                    None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank] * rows
            hor_con = [imageBlank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale,
                                             scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(imgArray)
            ver = hor
        return ver

    def empty(a):
        pass

    # =[130, 180, 0, 150, 0, 255]
    # path = 'image_ori/11.bmp'
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars",680,240)
    # cv2.createTrackbar("Hue min", "TrackBars", 0, 179, empty)
    # cv2.createTrackbar("Hue max", "TrackBars", 179, 179, empty)
    # cv2.createTrackbar("Sat min", "TrackBars", 0, 255, empty)
    # cv2.createTrackbar("Sat max", "TrackBars", 255, 255, empty)
    # cv2.createTrackbar("Val min", "TrackBars", 155, 255, empty)
    # cv2.createTrackbar("Val max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("Hue min", "TrackBars", 20, 180, empty)
    cv2.createTrackbar("Hue max", "TrackBars", 68, 180, empty)
    cv2.createTrackbar("Sat min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("Sat max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("Val min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("Val max", "TrackBars", 255, 255, empty)
    # 工具条上最大最小值
    # path_="5.jpeg"
    # img = cv2.imread(path_)
    img = cv2.imdecode(np.fromfile(path_, dtype=np.uint8), 1)
    while True:
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 根据字符取数据值
        h_min = cv2.getTrackbarPos("Hue min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val max", "TrackBars")
        # print(h_min, h_max, s_min, s_max, v_min, v_max)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHSV, lower, upper)
        imgResult = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow("original", img)
        # cv2.imshow("HSV",imgHSV)
        cv2.imshow("Mask", mask)
        # cv2.imshow("imgResult",imgResult)
        #  imgStack=stackImages(0.3,([img,imgHSV],[mask,imgResult]))
        #  cv2.imshow("stacked images",imgStack)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    cv2.destroyAllWindows()
if __name__ == '__main__':
    root = Tk()
    path = StringVar()
    Label(root, text="图片路径:").grid(row=0, column=0)
    Entry(root, textvariable=path).grid(row=0, column=1)
    Button(root, text="打开图片", command=selectPath).grid(row=0, column=2)
    Label(root, text="备注：按q退出图片显示").grid(row=1, column=0)
    root.mainloop()