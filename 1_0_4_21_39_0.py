from PIL import Image
import cv2
import numpy as np
import os

# 提取图中轮廓的个数
def ShapeDetection(path):
    contours,hierarchy = cv2.findContours(path,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #寻找轮廓点
    # print(len(contours))
    info=dict()
    for obj in contours:
        area = cv2.contourArea(obj)  # 计算轮廓内区域的面积
        # cv2.drawContours(imgContour, obj, -1, (255, 0, 0), 4)  # 绘制轮廓线
        perimeter = cv2.arcLength(obj, True)  # 计算轮廓周长
        approx = cv2.approxPolyDP(obj, 0.02 * perimeter, True)  # 获取轮廓角点坐标
        CornerNum = len(approx)  # 轮廓角点的数量
        x, y, w, h = cv2.boundingRect(approx)  # 获取坐标值和宽度、高度
        info[area]=[x,y,w,h]
    info_sorted=sorted(info.items(),reverse=True)
    return info_sorted[:3]
"""
提取图中的颜色部分
"""
def ColorDetection(img,color_hsv):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_hsv=color_hsv[0]
    high_hsv=color_hsv[1]
    low_hsv = np.array(low_hsv)
    high_hsv = np.array(high_hsv)
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    return mask
def detection_21_39(src,red_hsv=[[0, 46, 0], [180, 255, 255]]):
    mask = ColorDetection(src, red_hsv)
    info_sorted = ShapeDetection(mask)
    state=image_detection(src, info_sorted)
    return state
## 断路器状态识别 以左下角红色为基准 top识别(未)储能状态 rtop识别开合闸状态
def image_detection(src,info_sorted):
    max=10000
    ## 选出红色正方形
    for k,v in info_sorted:
        x,y,w,h=v
        im=src[y:y+h,x:x+w]
        b, g, r = cv2.split(im)
        if(np.mean(r)>np.mean(g) and np.mean(r)>np.mean(b) and abs(1-abs(w/h))<max):
            max=abs(1-abs(w/h))
            red_x,red_y,red_w,red_h=v
    red=red_w if red_w>red_h else red_h
    top_x,top_y,top_w,top_h=[red_x,int(red_y-1.28*red+1),red_w,int(0.456*red+1)]
    rtop_x,rtop_y,rtop_w,rtop_h=[int(red_x+1.78*red+1),int(red_y-1.28*red+1),red,int(0.456*red+1)]
    im_top=src[top_y:top_y+top_h,top_x:top_x+top_w]
    im_rtop=src[rtop_y:rtop_y+rtop_h,rtop_x:rtop_x+rtop_w]
    top_b, top_g, top_r = cv2.split(im_top)
    state=[0]*2
    if(np.mean(top_b)/(np.mean(top_b)+np.mean(top_g)+np.mean(top_r))>=0.3):
        # print('top:白色')
        state[1]=0
    else:
        # print('top:黄色')
        state[1]=1
    rtop_b,rtop_g,rtop_r=cv2.split(im_rtop)
    if(np.mean(rtop_r)>np.mean(rtop_b) and np.mean(rtop_r)>np.mean(rtop_g)):
        # print('rtop:红色')
        state[0]=0
    elif(np.mean(rtop_g)>np.mean(rtop_r) and np.mean(rtop_g)>np.mean(rtop_b)):
        # print('rtop:绿色')
        state[0]=1
    # cv2.imshow("top", im_top)
    # cv2.imshow("rtop",im_rtop)
    # cv2.waitKey(0)
    print(state)
    return state

if __name__=='__main__':
    rootDir='1_0_4_21_39_0'
    test_hsv = [[0, 46, 0], [180, 255, 255]]
    for filename in os.listdir(rootDir):
        pathname = os.path.join(rootDir, filename)
        # print(pathname)
        src = cv2.imread(pathname)
        detection_21_39(src, test_hsv)
        # cv2.imshow("shape detection", src)
        # cv2.imshow("mask",mask)
        # cv2.waitKey(0)

