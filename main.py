import numpy as np
import cv2
from utils import show, biggest_contour, four_point_transform, image_edges,order_points

path = "/images/original.jpg"
image = cv2.imread(path)
image = cv2.resize(image, (image.shape[1]//5,image.shape[0]//5))
edge = image_edges(image)

contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
biggest_cnt=biggest_contour(contours)
warped = four_point_transform(biggest_cnt, image)

cv2.drawContours(image,[biggest_cnt], -1, (0, 255, 0), 3)
show(image, "image0")
show(edge, "image1")
show(warped, "image2")

cv2.waitKey(0)
cv2.destroyAllWindows()







