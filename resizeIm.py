import cv2

im = cv2.imread('images/bmw.png')
im2 = cv2.resize(im, (0,0), fx=0.5, fy=0.5)

cv2.imwrite('images/bmw.jpg', im2)
