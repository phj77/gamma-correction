import cv2
import numpy as np
img_idx = 2
img = cv2.imread(f"img{img_idx}.jpg")

img = img.astype(np.float32) / 255.0

b = img[:, :, 0].astype(np.float32)
g = img[:, :, 1].astype(np.float32)
r = img[:, :, 2].astype(np.float32)

y = 0.299 * r + 0.587 * g + 0.114 * b

y_norm = np.log(1 + y) / np.log(1 + np.max(y))

ret, thresh_b = cv2.threshold(y_norm, 0.5, 1, cv2.THRESH_BINARY)
thresh_d =  1 - thresh_b

y_b = y_norm * thresh_b
y_d = y_norm * thresh_d

# dark side gamma correction
r_init = 1
r_cur = r_init
r_before = r_cur + 100 # 첫 번째 루프로 진입하기 위해 큰 값으로 설정

s = y_d[y_d != 0]
log_y_d = np.log(s)
std_d = np.std(s)

for i in range(100):
    s_r = s ** r_cur
    mean_s_log = np.mean(s_r * log_y_d)
    up = np.mean(s_r) - std_d
    r_next = r_cur - up/mean_s_log
    #print(up)

    r_before = r_cur
    r_cur = r_next

    if np.abs(r_cur - r_before) < 1e-7:
        break
#r_cur = np.clip(r_cur, 1e-6, 1.0)
r_d = r_cur

print("dark side gamma:", r_d)
# bright side gamma correction
r_init = 1
r_cur = r_init
r_before = r_cur + 100 # 첫 번째 루프로 진입하기 위해 큰 값으로 설정

s = y_b[y_b != 0]
log_y_b = np.log(s)
std_b = 1 - std_d

for i in range(100):
    s_r = s ** r_cur
    mean_s_log = np.mean(s_r * log_y_b)
    up = np.mean(s_r) - std_b
    r_next = r_cur - up/mean_s_log

    r_before = r_cur
    r_cur = r_next

    if np.abs(r_cur - r_before) < 1e-7:
        break

#r_cur = np.clip(r_cur, 1.0, 10.0)
r_b = r_cur

print("bright side gamma:", r_b)

enhanced_y_d = y_norm ** r_d
enhanced_y_b = y_norm ** r_b

cv2.imshow('Enhanced Y Dark', enhanced_y_d)
cv2.imshow('Enhanced Y Bright', enhanced_y_b)
cv2.waitKey(0)
cv2.destroyAllWindows()

sigma = 0.5
w = np.exp(-1 / (sigma ** 2 * 2) * (enhanced_y_b ** 2))
y_o = w * enhanced_y_d + (1 - w) * enhanced_y_b

cv2.imshow('Enhanced Y', y_o)
cv2.waitKey(0)
cv2.destroyAllWindows()
#y_o = y_b ** r_b + y_d ** r_d

# satur_d = 1 - np.tanh(y_d)
# satur_d[satur_d == 1] = 0
# satur_b = 1 - np.tanh(y_b)
# satur_b[satur_b == 1] = 0
# satur = satur_d + satur_b

satur = 1 - np.tanh(enhanced_y_b)

for i in range(0, 3):
    img[:,:,i] = (img[:,:,i] / (y_norm + 1e-8)) ** satur * y_o

img_uint8 = (img * 255).astype(np.uint8)
cv2.imwrite(f'enhanced_image_{img_idx}.jpg', img_uint8)
cv2.imshow('Enhanced Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()