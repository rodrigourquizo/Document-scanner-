import numpy as np
import cv2

def show(image,name):
    if type(image) == str:
        image = cv2.imread(image)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 500,500)
    cv2.imshow(name, image)

def image_edges(image):
    if type(image) == str:
        image = cv2.imread(image)
    sigma = 0.33
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def biggest_contour(contours):
    biggest = 0
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.1 * peri, True)
        if area > max_area:
            max_area = area
            biggest = approx
    return biggest

def order_points(pts):
    pts = pts.reshape(4,2)
    rect = np.zeros((4, 2))
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(points, image):
    rect = order_points(points).astype("float32")
    (zero, one, two, three) = rect

    width_1 = np.sqrt((zero[0]-one[0])**2 + (zero[1]-one[1])**2)
    width_2 = np.sqrt((three[0]-two[0])**2 + (three[1]-two[1])**2)
    max_width = int(max(width_2, width_1))

    height_1 = np.sqrt((zero[0]-three[0])**2 + (zero[1]-three[1])**2)
    height_2 = np.sqrt((one[0] - two[0])**2 + (one[0] - two[0])**2)
    max_height = int(max(height_1, height_2))

    destination = np.array([[0,0], [max_width, 0], [max_width,max_height], [0,max_height]], dtype='float32')

    M = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped
