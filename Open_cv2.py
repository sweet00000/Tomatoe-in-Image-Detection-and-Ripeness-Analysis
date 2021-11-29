import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
#x = input("Enter file path:")
#y = input("Sort by 'area', 'x', 'y', 'R':")
#z = input("Name of csv:")
os.chdir("C:/users/Sweet/source/repos/assets")
img = cv2.imread("C:/users/Sweet/source/repos/assets/0test.jpg", 1)
D = np.zeros(8)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (0,0,85), (50,255,255))
res = cv2.bitwise_and(img,img, mask= mask)
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for n in contours:
    if cv2.contourArea(n) >30000:
        x,y,w,h = cv2.boundingRect(n)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        print(np.array(cv2.mean(hsv[y:y+h,x:x+w])).astype(np.uint8))
        cv2.imshow("img", img[y:y+h,x:x+w])
        #cv2.waitKey(0)
        a = np.array(( n, (int(cv2.contourArea(n))), x, y), dtype=object)
        b = np.array(cv2.mean(hsv[y:y+h,x:x+w])).astype(np.uint8)
        c = np.concatenate((a,b))
        D = np.vstack((D, c))
cv2.destroyAllWindows()
cv2.imwrite('TEST.jpg', img)
img = cv2.resize(img, None, fx=1/3, fy=1/3)
d = pd.DataFrame(D, columns = ['cnt', 'area', 'x', 'y', 'h', 's', 'v', '0'])
d = d.drop(0, axis = 0)
d['h'] = (d['h']*(360/179)).astype(int)
d['s'] = (d['s']/1.79).astype(int)
d['v'] = (d['v']/1.79).astype(int)
d['R'] = ['ripe' if x<48 else 'unripe' for x in d['h']]
d['h'] = d['h'].astype(str)
d['s'] = d['s'].astype(str)
d['v'] = d['v'].astype(str)
d['hsv'] = d[['h', 's', 'v']].agg(' '.join, axis=1)
d = d[['area', 'x', 'y', 'hsv', 'R']]
d = d.sort_values(by=['x'])
d.to_csv("TEST.csv", index = None)
cv2.imshow("selected", img)
os.startfile("C:/users/Sweet/source/repos/assets/test.csv")
cv2.waitKey(0)
cv2.destroyAllWindows()