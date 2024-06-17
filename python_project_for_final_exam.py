
import os
import cv2 as cv
import numpy as np
import time
import math
import random
import datetime
Events = [i for i in dir(cv) if 'EVENT' in i]

#偵測按鍵


def click_event(event, x, y, flags, param):
    global mousex,mousey
    if event == cv.EVENT_LBUTTONDOWN:
        print(x,",",y)
        mousex = x
        mousey = y
        cv.destroyAllWindows()



#中值濾波rgb(5x5)

def med_filterRGB5x5(target_image, color_format):
    nr,nc = target_image.shape[:2]
    sample_image = target_image.copy()
    filter_list = np.zeros((5,5,3), dtype='uint8')
    compare = []
    print(color_format)
    if color_format == "rgb":
        processed_image = np.zeros((nr,nc,3), dtype='uint8')
    else:
        processed_image = np.zeros((nr,nc), dtype='uint8')
    for x in range(nr):
        for y in range(nc):
            for i in range(-2,3):
                for j in range(-2,3):
                    try:
                        filter_list[1+i,1+j] = sample_image[x+i,y+j][0:3]
                    except IndexError:
                        filter_list[1+i,1+j] = int(np.mean(filter_list))
                        continue
            print(x,y)
            colors = np.median(np.median(filter_list, axis=0), axis=0)
            processed_image[x,y] = colors
            #for a1,a2,a3,a4,a5 in filter_list:
                #compare.append(a1)
                #compare.append(a2)
                #compare.append(a3)
                #compare.append(a4)
                #compare.append(a5)
            
            #if color_format == "rgb":
                #color_b = []
                #color_g = []
                #color_r = []
                #for b,g,r in compare:
                    #color_b.append(b)
                    #color_g.append(g)
                    #color_r.append(r)
                #color_b.sort()
                #color_g.sort()
                #color_r.sort()
                #colors = [color_b[12], color_g[12], color_r[12]]
                #processed_image[x,y] = colors
        print("processed line",x)
    return processed_image

#中值濾波rgb

def med_filterRGB(target_image, color_format):
    nr,nc = target_image.shape[:2]
    sample_image = target_image.copy()
    filter_list = np.zeros((3,3,3), dtype='uint8')
    compare = [0,0,0,0,0,0,0,0,0]
    if color_format == "rgb":
        processed_image = np.zeros((nr,nc,3), dtype='uint8')
    else:
        processed_image = np.zeros((nr,nc), dtype='uint8')
    for x in range(nr):
        for y in range(nc):
            for i in range(-1,2):
                for j in range(-1,2):
                    try:
                        filter_list[1+i,1+j] = sample_image[x+i,y+j][0:3]
                    except IndexError:
                        continue

            compare[0],compare[1],compare[2],compare[3],compare[4],compare[5],compare[6],compare[7],compare[8] = filter_list[0,0],filter_list[0,1],filter_list[0,2],filter_list[1,0],filter_list[1,1],filter_list[1,2],filter_list[2,0],filter_list[2,1],filter_list[2,2]
            if color_format == "rgb":
                color_b = []
                color_g = []
                color_r = []
                for b,g,r in compare:
                    color_b.append(b)
                    color_g.append(g)
                    color_r.append(r)
                color_b.sort()
                color_g.sort()
                color_r.sort()
                colors = [color_b[4], color_g[4], color_r[4]]
                processed_image[x,y] = colors
        print("processed line",x)
    return processed_image

#中值濾波灰階

def med_filterGray(target_image, color_format):
    nr,nc = target_image.shape[:2]
    target_image = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)
    sample_image = target_image.copy()
    filter_list = np.zeros((3,3), dtype='uint8')
    compare = [0,0,0,0,0,0,0,0,0]
    if color_format == "rgb":
        processed_image = np.zeros((nr,nc,3), dtype='uint8')
    else:
        processed_image = np.zeros((nr,nc), dtype='uint8')
    for x in range(nr):
        for y in range(nc):
            for i in range(-1,2):
                for j in range(-1,2):
                    try:
                        if color_format == "rgb":
                            filter_list[1+i,1+j] = round(0.114 * sample_image[x+i,y+j][0] + 0.587 * sample_image[x+i,y+j][1] + 0.299 * sample_image[x+i,y+j][2])
                        else:
                            filter_list[1+i,1+j] = sample_image[x+i,y+j]
                    except IndexError:
                        continue

            compare[0],compare[1],compare[2],compare[3],compare[4],compare[5],compare[6],compare[7],compare[8] = filter_list[0,0],filter_list[0,1],filter_list[0,2],filter_list[1,0],filter_list[1,1],filter_list[1,2],filter_list[2,0],filter_list[2,1],filter_list[2,2]
            data = compare.copy()
            compare.sort()
            loc = [data.index(compare[5]) // 3,data.index(compare[5]) % 3]
            locx,locy = loc[0],loc[1]
            
            try:
                processed_image[x,y] = target_image[x-1+locx,y-1+locy][0:3]
            except IndexError:
                processed_image[x,y] = target_image[x,y][0:3]
        print("processed line",x)
    return processed_image

#轉灰階

def grey_scale(img, color_format):
    if type(img[0,0]) == np.uint8:
        return img
    else:
        img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img1

#依色彩比例調色

def color_ratio(img1):
    nr,nc = img1.shape[:2]
    processed = np.zeros([nr,nc,3],dtype='uint8')
    if type(img1[0,0]) == np.uint8:
        img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    print("Enter the ratio of r,g,b")
    r = eval(input("r(0~255):"))
    g = eval(input("g(0~255):"))
    b = eval(input("b(0~255):"))
    for x in range(nr):
        for y in range(nc):
            processed[x,y][0] = np.around(img1[x,y][0] * b / 255)
            processed[x,y][1] = np.around(img1[x,y][1] * g / 255)
            processed[x,y][2] = np.around(img1[x,y][2] * r / 255)
    return processed

#混合模式

def blend(img1,img2):
    if type(img1[0,0]) == np.uint8:
        img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
        print("converted")
    if type(img2[0,0]) == np.uint8:
        img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
        print("converted")
    print(img1.shape)
    print(img2.shape)
    nr,nc = img1.shape[:2]
    img3 = np.zeros([nr,nc,3], dtype='uint8')
    print(len(img3[0,0]))
    while True:
        print("-------------------------------------------------")
        print("1\t\t\t\t\tcircular fade")
        print("2\t\t\t\t\tfade using cloud noise texture")
        print("3\t\t\t\t\tblend using checker texture")
        print("4\t\t\t\t\tblend using custom texture")
        print("5\t\t\t\t\tcube fade")
        print("6\t\t\t\t\trectangular fade")
        print("7\t\t\t\t\tplus")
        print("8\t\t\t\t\tminus")
        print("-------------------------------------------------")
    
        mode = input("Choose the blending mode:")

        #圓形漸淡

        if mode == "1":
            #y = eval(input("enter location x(int):"))
            #x = eval(input("enter location y(int):"))
            cv.imshow("select the position", img1)
            cv.setMouseCallback("select the position", click_event)
            cv.waitKey(0)
            cv.destroyAllWindows()
            max_dist = eval(input("enter max distance(int):"))
            min_dist = eval(input("enter min distance(int):"))
            diff = max_dist - min_dist
            print("processing...")
            for fx in range(nr):
                for fy in range(nc):
                    point_dist = math.sqrt((fx-mousey) ** 2 + (fy-mousex) ** 2) 
                    if point_dist <= min_dist:
                        img3[fx,fy] = img2[fx,fy][0:3]
                    elif point_dist >= max_dist:
                        img3[fx,fy] = img1[fx,fy][0:3]
                    else:
                        ratio = (point_dist - min_dist) / diff
                        img3[fx,fy] = (1-ratio) * img2[fx,fy][0:3] + ratio * img1[fx,fy][0:3]
            print("completed")
            return img3

        #雲雜訊漸淡

        elif mode == '2':
            maxnum = eval(input("Enter maximum intensity(0~255):"))
            minnum = eval(input("Enter minimum intensity(0~255):"))
            noise = np.random.randint(minnum, maxnum, size=(nr,nc), dtype='uint8')
            ranlocy = random.randint(0, nr-10)
            ranlocx = random.randint(0, nc-17)
            cloudnoise = cv.resize(noise[ranlocy:ranlocy+9,ranlocx:ranlocx+16], [nc,nr])
            print("processing...")
            for fx in range(nr):
                for fy in range(nc): 
                    img3[fx,fy] = np.round(cloudnoise[fx,fy] / 255 * img2[fx,fy] + (1-cloudnoise[fx,fy]/255) * img1[fx,fy])
            print("completed")
            return img3

        #棋盤材質漸淡

        elif mode == '3':
            block_size = eval(input("enter block size(int):"))
            blocky = nr // block_size+1
            blockx = nc // block_size+1
            blocks = np.zeros([nr,nc], dtype='uint8')
            print("processing...")
            for gx in range(blockx):
                for gy in range(blocky):
                    for fx in range(block_size):
                        for fy in range(block_size):
                            try:
                                blocks[fx+gy*block_size,fy+gx*block_size] = (1 - ((gx+gy) % 2)) * 255
                            except IndexError:
                                continue
            for fx in range(nr):
                for fy in range(nc): 
                    img3[fx,fy] = np.round(blocks[fx,fy] / 255 * img2[fx,fy] + (1-blocks[fx,fy]/255) * img1[fx,fy])
            print("completed")
            return img3

        #自訂濾鏡檔案路徑以進行混合

        elif mode == '4':
            print("------instruction------")
            print("0 --> 100% first image")
            print("255 --> 100% second image")
            print("-----------------------")
            customfilter_location = input("Enter the file path of custom filter:")
            customfilter = cv.imread(customfilter_location, -1)
            customfilter = cv.resize(customfilter, [nc,nr])
            if type(customfilter[0,0]) != np.uint8:
                customfilter = cv.cvtColor(customfilter, cv.COLOR_BGR2GRAY)
            for fx in range(nr):
                for fy in range(nc):
                    img3[fx,fy] = np.round(customfilter[fx,fy] / 255 * img2[fx,fy] + (1-customfilter[fx,fy]/255) * img1[fx,fy])
            return img3

        #方形漸淡

        elif mode == '5':
            cv.imshow("select the position", img1)
            cv.setMouseCallback("select the position", click_event)
            cv.waitKey(0)
            cv.destroyAllWindows()
            max_dist = eval(input("enter max distance(int):"))
            min_dist = eval(input("enter min distance(int):"))
            diff = max_dist - min_dist
            print("processing...")
            for fx in range(nr):
                for fy in range(nc):
                    if math.sqrt((fx-mousey)**2) > math.sqrt((fy-mousex)**2):
                        point_dist = math.sqrt((fx-mousey)**2)
                    else:
                        point_dist = math.sqrt((fy-mousex)**2)
                    if point_dist <= min_dist:
                        img3[fx,fy] = img2[fx,fy][0:3]
                    elif point_dist >= max_dist:
                        img3[fx,fy] = img1[fx,fy][0:3]
                    else:
                        ratio = (point_dist - min_dist) / diff
                        img3[fx,fy] = (1-ratio) * img2[fx,fy][0:3] + ratio * img1[fx,fy][0:3]
            print("completed")
            return img3

        #矩形漸淡

        elif mode == '6':
            cv.imshow("select the position", img1)
            cv.setMouseCallback("select the position", click_event)
            cv.waitKey(0)
            cv.destroyAllWindows()
            fade_dist = eval(input("enter fade distance(int):"))
            miny_dist = eval(input("enter min distance for y(int):"))
            minx_dist = eval(input("enter min distance for x(int):"))

            
            
            print("processing...")
            for fx in range(nr):
                for fy in range(nc):
                    disty = (fx-mousey)
                    distx = (fy-mousex)
                    if fade_dist != 0:
                        if disty > miny_dist and distx > minx_dist:
                            point_dist = math.sqrt((fx-(mousey+miny_dist))**2 + (fy-(mousex+minx_dist))**2)
                            ratio = point_dist / fade_dist
                            if ratio >= 1:
                                img3[fx,fy] = img1[fx,fy][0:3]
                            else:
                                img3[fx,fy] = (1-ratio) * img2[fx,fy][0:3] + ratio * img1[fx,fy][0:3]
                        elif disty > miny_dist and distx < (-minx_dist):
                            point_dist = math.sqrt((fx-(mousey+miny_dist))**2 + (fy-(mousex-minx_dist))**2)
                            ratio = point_dist / fade_dist
                            if ratio >= 1:
                                img3[fx,fy] = img1[fx,fy][0:3]
                            else:
                                img3[fx,fy] = (1-ratio) * img2[fx,fy][0:3] + ratio * img1[fx,fy][0:3]
                        elif disty < (-miny_dist) and distx < (-minx_dist):
                            point_dist = math.sqrt((fx-(mousey-miny_dist))**2 + (fy-(mousex-minx_dist))**2)
                            ratio = point_dist / fade_dist
                            if ratio >= 1:
                                img3[fx,fy] = img1[fx,fy][0:3]
                            else:
                                img3[fx,fy] = (1-ratio) * img2[fx,fy][0:3] + ratio * img1[fx,fy][0:3]
                        elif disty < (-miny_dist) and distx > minx_dist:
                            point_dist = math.sqrt((fx-(mousey-miny_dist))**2 + (fy-(mousex+minx_dist))**2)
                            ratio = point_dist / fade_dist
                            if ratio >= 1:
                                img3[fx,fy] = img1[fx,fy][0:3]
                            else:
                                img3[fx,fy] = (1-ratio) * img2[fx,fy][0:3] + ratio * img1[fx,fy][0:3]
                        else:
                            if math.sqrt(disty**2) > miny_dist:
                                point_dist = math.sqrt((math.sqrt(disty**2) - miny_dist)**2)
                                ratio = point_dist / fade_dist
                                if ratio >= 1:
                                    img3[fx,fy] = img1[fx,fy][0:3]
                                else:
                                    img3[fx,fy] = (1-ratio) * img2[fx,fy][0:3] + ratio * img1[fx,fy][0:3]
                            elif math.sqrt(distx**2) > minx_dist:
                                point_dist = math.sqrt((math.sqrt(distx**2) - minx_dist)**2)
                                ratio = point_dist / fade_dist
                                if ratio >= 1:
                                    img3[fx,fy] = img1[fx,fy][0:3]
                                else:
                                    img3[fx,fy] = (1-ratio) * img2[fx,fy][0:3] + ratio * img1[fx,fy][0:3]
                            else:
                                img3[fx,fy] = img2[fx,fy][0:3]
                    else:
                        if math.sqrt((fx-mousey)**2) < miny_dist and math.sqrt((fy-mousex)**2) < minx_dist:
                            img3[fx,fy] = img2[fx,fy][0:3]
                        else:
                            img3[fx,fy] = img1[fx,fy][0:3]
            print("completed")
            return img3

        #相加

        elif mode == '7':
            for x in range(nr):
                for y in range(nc):
                    for c in range(3):
                        if (int(img1[x,y,c]) + int(img2[x,y,c])) >= 255:
                            img3[x,y,c] = 255
                        else:
                            img3[x,y,c] = img1[x,y,c] + img2[x,y,c]
            return img3

        #相減

        elif mode == '8':
            for x in range(nr):
                for y in range(nc):
                    for c in range(3):
                        if (int(img1[x,y,c]) - int(img2[x,y,c])) <= 0:
                            img3[x,y,c] = 0
                        else:
                            img3[x,y,c] = img1[x,y,c] - img2[x,y,c]
            return img3

#對比(使用線性)

def contrast(img1):
    if type(img1[0,0]) == np.uint8:
        img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
        print("converted")
    nr,nc = img1.shape[:2]
    processed = np.zeros([nr,nc,3], dtype='uint8')
    value = eval(input("Enter the intensity of contrast(0~127):"))
    max_n = 255 - value
    min_n = 0 + value
    for x in range(nr):
        for y in range(nc):
            for i in range(3):
                if img1[x,y][i] < min_n:
                    color = 0
                elif img1[x,y][i] > max_n:
                    color = 255
                else:
                    diff = max_n - min_n
                    color = np.around((img1[x,y][i] - min_n) / diff * 255)
                processed[x,y][i] = color
    cv.imshow("processed", processed)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return processed

#色彩盤

def create_color_wheel(radius):
    diameter = radius * 2
    color_wheel = np.zeros((diameter, diameter, 3), dtype=np.uint8)

    center = (radius, radius)
    for y in range(diameter):
        for x in range(diameter):
            dx = x - center[0]
            dy = y - center[1]
            r = np.sqrt(dx**2 + dy**2)
            if r <= radius:
                theta = np.arctan2(dy, dx)
                hue = ( (theta + np.pi) / (2 * np.pi) ) -0.00001
                sat = r / radius
                val = 255
                color_wheel[y, x] = hsv_to_bgr(hue, sat, val)

    return color_wheel

#hsv轉bgr

def hsv_to_bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0

    if h_i == 0: r, g, b = v, t, p
    if h_i == 1: r, g, b = q, v, p
    if h_i == 2: r, g, b = p, v, t
    if h_i == 3: r, g, b = p, q, v
    if h_i == 4: r, g, b = t, p, v
    if h_i == 5: r, g, b = v, p, q

    return (b, g, r)  # OpenCV uses BGR color space

#黑白漸層圖

def create_brightness(size):
    brightness = np.zeros([size,size], dtype='uint8')
    for x in range(size):
        for y in range(size):
            brightness[x,y] = np.around(y / size * 255)
    return brightness

#自由畫線

def draw_line(img1):
    global color_circle, radius_circle
    
    def line_drawing(event,x,y,flags,param):
        global pt1_x,pt1_y,drawing
        
        if event==cv.EVENT_LBUTTONDOWN:
            drawing=True
            pt1_x,pt1_y=x,y

        elif event==cv.EVENT_MOUSEMOVE:
            if drawing==True:
                cv.line(img,(pt1_x,pt1_y),(x,y),color=color_circle,thickness=radius_circle)
                pt1_x,pt1_y=x,y

        elif event==cv.EVENT_LBUTTONUP:
            drawing=False

        pt1_x,pt1_y=x,y

    img = img1.copy()
    
    
    colorw = create_color_wheel(300)
    print("Select the color")
    cv.imshow("select the color", colorw)
    cv.setMouseCallback("select the color", click_event)
    cv.waitKey(0)
    cv.destroyAllWindows()
    color = colorw[mousey,mousex]
    print(color)
    brightw = create_brightness(300)
    print("Select the brightness")
    cv.imshow("select the brightness", brightw)
    cv.setMouseCallback("select the brightness", click_event)
    cv.waitKey(0)
    cv.destroyAllWindows()
    brightness = brightw[mousey,mousex]
    print(brightness)
    color_circle = color / 255 * brightness
    print(color_circle)
    radius_circle = eval(input("Enter radius:"))
    drawing=False
    print("----------------------")
    print("Left Click--------Draw")
    print("esc---------------Exit")
    print("----------------------")
    time.sleep(1)
    cv.namedWindow('test draw')
    cv.setMouseCallback('test draw',line_drawing)
    while(1):
        cv.imshow('test draw',img)
        k=cv.waitKey(1)&0xFF
        if k==27:
            break
    cv.destroyAllWindows()
    return img

#卷積運算

def kernel_cal(img1):
    kernel = np.array([[0,0,0],[0,0,0],[0,0,0]], dtype='int8')
    for x in range(3):
        for y in range(3):
            kernel[x][y] = eval(input(f'Enter kernel[{x}][{y}]'))
    img2 = cv.filter2D(img1, -1, kernel)
    return img2

#主程式

def main():
    #選取檔案
    fileloc = input("Enter the image file path:")
    img1 = cv.imread(fileloc, -1)
    img2 = None
    tempimg = None

    #選擇功能

    while True:
        print("\033[H\033[J", end="")
        if type(img1[0,0]) == np.uint8:
            img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
        elif type(img1[0,0]) == np.ndarray:
            if len(img1[0,0]) != 3:
                img1 = cv.cvtColor(img1, cv.COLOR_BGRA2BGR)
        print("-----------------features------------------")
        print("1\t\t\t\t\t\tChange color ratio")
        print("2\t\t\t\t\t\tContrast")
        print("3\t\t\t\t\t\tDraw")
        print("4\t\t\t\t\t\tKernel Calculation(3x3)")
        print("5\t\t\t\t\t\tGrayscale")
        print("6\t\t\t\t\t\tMedian Filter")
        print("7\t\t\t\t\t\tAuto Binarization")
        print("8\t\t\t\t\t\tBlend")
        print("9\t\t\t\t\t\tReverse Blend")
        print("-------------------------------------------")
        imgr,imgc = img1.shape[:2]
        mode = input("Enter the mode number:")

        #導向功能函式

        if mode == '1':
            tempimg = color_ratio(img1)
        elif mode == '2':
            tempimg = contrast(img1)
        elif mode == '3':
            tempimg = draw_line(img1)
        elif mode == '4':
            tempimg = kernel_cal(img1)
        elif mode == '5':
            tempimg = grey_scale(img1, 'rgb')
        elif mode == '6':

            #選擇中值濾波器

            print("\033[H\033[J", end="")
            print("-----------------modes------------------")
            print("1\t\t\t\t\t\tgrayscale")
            print("2\t\t\t\t\t\trgb(3x3)")
            print("3\t\t\t\t\t\trgb(5x5)")
            print("----------------------------------------")
            filtermode = input("Enter the mode number:")
            if filtermode == '1':
                tempimg = med_filterGray(img1, 'gray')
            elif filtermode == '2':
                tempimg = med_filterRGB(img1, 'rgb')
            elif filtermode == '3':
                tempimg = med_filterRGB5x5(img1, 'rgb')
            else:
                tempimg = np.zeros([3,3,3], dtype='uint8')
        elif mode == '7':
            gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            size = eval(input("Enter the detecting area size(3 or higher):"))
            tempimg = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, size, -30)
        elif mode == '8':
            print("\033[H\033[J", end="")
            if img2 == None:
                fileloc = input("Enter the second image file path:")
                tempimg = cv.imread(fileloc, -1)
                tempimg = cv.resize(tempimg, [imgc,imgr])
                if type(tempimg[0,0]) == np.uint8:
                    tempimg = cv.cvtColor(tempimg, cv.COLOR_GRAY2BGR)
                elif type(tempimg[0,0]) == np.ndarray:
                    if len(tempimg[0,0]) != 3:
                        tempimg = cv.cvtColor(tempimg, cv.COLOR_BGRA2BGR)
                
                tempimg = blend(img1, tempimg)
        elif mode == '9':
            print("\033[H\033[J", end="")
            if img2 == None:
                fileloc = input("Enter the second image file path:")
                tempimg = cv.imread(fileloc, -1)
                tempimg = cv.resize(tempimg, [imgc,imgr])
                if type(tempimg[0,0]) == np.uint8:
                    tempimg = cv.cvtColor(tempimg, cv.COLOR_GRAY2BGR)
                elif type(tempimg[0,0]) == np.ndarray:
                    if len(tempimg[0,0]) != 3:
                        tempimg = cv.cvtColor(tempimg, cv.COLOR_BGRA2BGR)
                

                tempimg = blend(tempimg, img1)
        else:
            continue
        
        cv.imshow("processed image", tempimg)
        cv.waitKey(0)
        cv.destroyAllWindows()
        print("\033[H\033[J", end="")

        #詢問是否混合上一張圖

        blend_bool = input("Blend with last image? (y/n):")
        if blend_bool == 'y':
            tempimg2 = tempimg.copy()
            cv.imwrite("blend_temp"+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".png", tempimg2)
            tempimg = blend(img1, tempimg)
            img1 = tempimg2.copy()
            cv.imshow("processed image", tempimg)
            cv.waitKey(0)
            cv.destroyAllWindows()
            print("\033[H\033[J", end="")

        #詢問是否反混合上一張圖

        blend_bool = input("Reverse blend with last image? (y/n):")
        if blend_bool == 'y':
            tempimg2 = tempimg.copy()
            cv.imwrite("blend_temp"+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".png", tempimg2)
            tempimg = blend(tempimg, img1)
            img1 = tempimg2.copy()
            cv.imshow("processed image", tempimg)
            cv.waitKey(0)
            cv.destroyAllWindows()
            print("\033[H\033[J", end="")

        #詢問是否儲存目前進度(處理後的混合詢問時如答是，則會自動儲存)
  
        save = input("Save changes? (y/n):")
        if save == 'y':
            img1 = tempimg
            img2 = None
            tempimg = None
            tempimg2 = None
        elif save == 'n':
            img1 = img1
            img2 = None
            tempimg = None
            tempimg2 = None

        #詢問是否繼續編輯

        exit_bool = input("Continue? (y/n):")
        if exit_bool == 'y':
            continue
        elif exit_bool == 'n':
            break

    #輸出結果到同一個資料夾

    cv.imwrite("output"+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".png", img1)
    print("Saved image as 'output.png'")
    print()
    print("System will close in 5 seconds...")
    time.sleep(5)


if __name__ == "__main__":
    main()

#unused code(connective component)
'''
clicked = False
x_coord, y_coord = -1, -1
def draw_connected_components(image_path):
    
    image = cv2.imread(image_path)

    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    binary = cv2.adaptiveThreshold(gray,128, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -30)

    cv2.imshow("a", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    num_labels, labels_im = cv2.connectedComponents(binary)

   
    output_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

   
    for label in range(1, num_labels):  
        
        component_mask = (labels_im == label).astype("uint8") * 255
        
       
        color = [random.randint(0, 255) for _ in range(3)]
        
       
        output_image[component_mask == 255] = color

  
    cv2.imshow('Connected Components', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_coordinates(event, x, y, flags, param):
    global clicked, x_coord, y_coord
    if event == cv2.EVENT_LBUTTONDOWN:
        x_coord, y_coord = x, y
        clicked = True


image = cv2.imread('Enako.jpg')

draw_connected_components('Enako.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


num_labels, labels_im = cv2.connectedComponents(binary)


cv2.imshow('Image', image)
cv2.setMouseCallback('Image', get_coordinates)


while not clicked:
    cv2.waitKey(1)

cv2.destroyAllWindows()

def get_segment_at_coordinates(x, y, labels_im, image):

    label = labels_im[y, x]
    
    if label == 0:
        print("The coordinates are in the background.")
        
        
        component_mask = np.ones_like(labels_im, dtype=np.uint8) * 255 
        component_mask[labels_im != 0] = 0  
        

        component = cv2.bitwise_and(image, image, mask=component_mask)
        return component


    component_mask = (labels_im == label).astype("uint8") * 255


    component = cv2.bitwise_and(image, image, mask=component_mask)

    return component

segment = get_segment_at_coordinates(x_coord, y_coord, labels_im, image)


if segment is not None:
    cv2.imshow('Segment at Coordinates', segment)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No segment found at the given coordinates.")
'''