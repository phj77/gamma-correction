import cv2
import numpy as np

def adjust_gamma(image, gamma=1.0):
    # 0~255 범위의 입력값에 대해 미리 계산된 결과 테이블 생성
    # 1.0 / gamma를 사용하는 이유는 보정의 방향성 때문입니다.
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")

    # LUT 함수를 사용하여 이미지의 픽셀값을 테이블에 맞게 변환
    return cv2.LUT(image, table)

# 이미지 로드
image = cv2.imread('img7.jpg')

# 감마 값 설정
# gamma > 1.0: 이미지가 밝아짐
# gamma < 1.0: 이미지가 어두워짐
bright_image = adjust_gamma(image, gamma=2.2)
#dark_image = adjust_gamma(image, gamma=0.5)

# 결과 확인
cv2.imshow("Original", image)
cv2.imshow("Gamma 2.2 (Bright)", bright_image)
#cv2.imshow("Gamma 0.5 (Dark)", dark_image)
cv2.waitKey(0)
cv2.destroyAllWindows()