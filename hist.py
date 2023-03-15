import cv2
import numpy as np


img_path = 'data/data1103/dark/rgb20.png'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
x = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
hist, bins = np.histogram(x[:, 0], bins=10)
hist = hist[1:]
bins = bins[1:]
hist_number = np.argmax(hist)
x = x[(x[:, 0] > bins[hist_number]) & (x[:, 0] < bins[hist_number + 1]), :]
mean_value = np.mean(x, axis=0).astype(np.uint8)
y = np.zeros((img.shape[0]*img.shape[1], img.shape[2]), dtype=np.uint8)
y[:, ] = mean_value
#lab转rgb再保存
y = cv2.cvtColor(y.reshape(img.shape[0], img.shape[1], img.shape[2]), cv2.COLOR_LAB2BGR)
cv2.imwrite('5.png', y.reshape(img.shape[0], img.shape[1], img.shape[2]))

