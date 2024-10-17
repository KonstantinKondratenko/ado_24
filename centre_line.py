import cv2
import numpy as np

def segment(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    cv2.imwrite("mask.png", mask)
    out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('./AFAFAF.png', out)
    return mask  # Возвращаем маску в формате CV_8UC1

def find_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours_and_centers(image, contours):
    centers = []
    for contour in contours:
        # Найдем центр контура
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
            # Рисуем центр контура зеленым цветом
            cv2.circle(image, (cX, cY), 5, (0, 255, 0), -1)

    # Сортируем центры по Y-координате (от нижней точки к верхней)
    centers.sort(key=lambda x: x[1])

    # Соединяем центры синей линией
    for i in range(len(centers) - 1):
        cv2.line(image, centers[i], centers[i + 1], (255, 0, 0), 2)

    cv2.imwrite('./contours_and_centers.png', image)
# Load the image
image = cv2.imread('./image_10.png')

# Segment the image
mask = segment(image)

contours = find_contours(mask)

# Draw contours and centers on the original image
draw_contours_and_centers(image, contours)
