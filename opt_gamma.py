import cv2
import numpy as np

img = cv2.imread('img2.jpg')

# 각 채널 분리 (BGR 순서)
b = img[:, :, 0].astype(np.float32)
g = img[:, :, 1].astype(np.float32)
r = img[:, :, 2].astype(np.float32)

# 가중치 합산으로 Y값 계산
y = 0.299 * r + 0.587 * g + 0.114 * b

y_norm = np.log(1 + y) / np.max(y)

ret, thresh_b = cv2.threshold(y_norm, 0.5, 1, cv2.THRESH_BINARY)
thresh_d =  1 - thresh_b

y_b = y * thresh_b
y_d = y * thresh_d

# dark side gamma correction
r_init = 1
r_cur = r_init
r_next = r_cur + 100

y_d_ = y_d
y_d_[y_d_ == 0] = 1
log_y_d = np.log(y_d_)

s = y_d.astype(np.float32)
flat_y_d = y_d.flatten()
flat_y_d = flat_y_d[flat_y_d != 0]
std_d = np.std(flat_y_d) 

while 0.1 ** 7 * -1 > r_next - r_cur or 0.1 ** 7 < r_next - r_cur: # 무한 루프 돌아감
    s_r = s ** r_cur
    sum_s = np.sum(s_r)
    sum_s_log = np.sum(s_r * log_y_d)
    r_next = r_cur - (sum_s - std_d)/sum_s_log
r_d = r_next

# bright side gamma correction
r_init = 10
r_cur = r_init
r_next = r_cur + 100

y_b_ = y_b
y_b_[y_b_ == 0] = 1
log_y_b = np.log(y_b_)

s = y_b.astype(np.float32)
flat_y_b = y_b.flatten()
flat_y_b = flat_y_b[flat_y_b != 0]
std_b = 1 - np.std(flat_y_d) 

while 0.1 ** 7 * -1 > r_next - r_cur or 0.1 ** 7 < r_next - r_cur:
    s_r = s ** r_cur
    sum_s = np.sum(s_r)
    sum_s_log = np.sum(s_r * log_y_b)
    r_next = r_cur - (sum_s - std_b)/sum_s_log
r_b = r_next

sigma = 0.5
w_d = np.exp(-1 * ( y_d ** 2 ) / (sigma ** 2 * 2))
w_b = np.exp(-1 * ( y_b ** 2 ) / (sigma ** 2 * 2))

y_o = w_d * y_d + (1 - w_b) * y_b

i_out = np.zeros(img.shape)
satur_d = 1 - np.tanh(y_d)
satur_d[satur_d == 1] = 0
satur_b = 1 - np.tanh(y_d)
satur_b[satur_b == 1] = 0
satur = satur_d + satur_b
for i in range(0, 3):
    img[:,:,i] = (img[:,:,i] / y) ** satur * y_o
