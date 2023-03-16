import cv2
import numpy as np


img_path = 'data/99/dark/rgb59.png'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

w = img.shape[0]
h = img.shape[1]

ratio = np.sqrt(5000 / (w * h))
ww, hh = int(ratio * w), int(ratio * h)
img = cv2.resize(img, (hh, ww))
x = img.reshape(img.shape[0]*img.shape[1], img.shape[2])

hist, bins = np.histogram(x[:, 0], bins=10)
hist = hist[1:]
bins = bins[1:]


sorted_indices = np.argsort(hist)
hist_number = sorted_indices[-1]
second_hist_number = sorted_indices[-2]

x = x[((x[:, 0] > bins[hist_number]) & (x[:, 0] < bins[hist_number + 1]))|((x[:, 0] > bins[second_hist_number]) & (x[:, 0] < bins[second_hist_number + 1])), :]

mean_value = np.mean(x, axis=0).astype(np.uint8)
y = np.zeros((img.shape[0]*img.shape[1], img.shape[2]), dtype=np.uint8)
y[:, ] = mean_value
#lab转rgb再保存
y = cv2.cvtColor(y.reshape(img.shape[0], img.shape[1], img.shape[2]), cv2.COLOR_LAB2BGR)
cv2.imwrite('59.png', y.reshape(img.shape[0], img.shape[1], img.shape[2]))

