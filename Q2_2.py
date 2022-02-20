import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt 
np.set_printoptions(threshold=sys.maxsize)

cap = cv.VideoCapture('Resources/ball_video2.mp4')

def binary_img_conversion(frame):
    gray_img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    _,binary_img = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY)

    return binary_img

def findpoints(frame):
    points = np.where(frame==0)
    x_cords = points[0]
    y_cords = points[1]
    ix_min = np.argmin(x_cords)
    ix_max = np.argmax(x_cords)
    cord_top = np.array([x_cords[ix_min], y_cords[ix_min]])
    cord_bottom = np.array([x_cords[ix_max], y_cords[ix_max]])
    
    return (cord_top, cord_bottom)

def convert_axis(cords):
    temp_cords = cords
    x_new = cords[:,1]
    y_new = 1676 - temp_cords[:,0]

    new_cords = np.vstack((x_new,y_new)).T
    return new_cords

def standard_least_sqr(cordinates):
    x_val = cordinates[:,0]
    y_val = cordinates[:,1]
    ones = np.ones(x_val.shape)
    zeros = np.vstack((np.square(x_val),x_val,ones)).T
    trans_1 = np.dot(zeros.transpose(),zeros)
    trans_2 = np.dot(np.linalg.inv(trans_1),zeros.transpose())
    A_matrix = np.dot(trans_2, y_val.reshape(-1,1))
    return A_matrix

def plot_parabola(coefficents, cordinates):
    x_val = cordinates[:,0]
    y_val = cordinates[:,1]
    x_min = np.min(x_val)
    x_max = np.max(x_val)
    x_plot = np.linspace(x_min-100,x_max+100,300)
    ones_plot = np.ones(x_plot.shape)
    z_plot = np.vstack((np.square(x_plot),x_plot, ones_plot)).T
    y_plot = np.dot(z_plot,coefficents)

    plt.title('LS Estimation for Video 2')
    plt.plot(x_val,y_val,'ro',x_plot,y_plot,'-b')
    plt.show()


plotarr = []
while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == False:
        break
    bin_frame = binary_img_conversion(frame)
    top,bottom = findpoints(bin_frame)
    plotarr.append(top)
    plotarr.append(bottom)

plotarr = np.array(plotarr)
plotarr = convert_axis(plotarr)
coeffcient_parabola = standard_least_sqr(plotarr)
plot_parabola(coeffcient_parabola,plotarr)

cv.waitKey(0)
cap.release()