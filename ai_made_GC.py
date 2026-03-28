import cv2
import numpy as np

def enhance_low_light_image(image_path):
    # 1. 이미지 로드 및 정규화
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("이미지를 불러올 수 없습니다.")
    
    # 컬러 복원시 범위를 맞추기 위해 0~1 사이의 float 형태로 변환
    img_float = img.astype(np.float32) / 255.0
    B, G, R = cv2.split(img_float)

    # 2. 휘도 계산 및 정규화 (Luminance Normalization)
    Y_i = 0.299 * R + 0.587 * G + 0.114 * B
    
    M = np.max(Y_i)
    # 0 나누기 방지 및 정규화 (논문의 수식 2 참조)
    Y_L = np.log(Y_i + 1.0) / np.log(M + 1.0)

    # 3. 최적의 감마 보정 파라미터 추정 (Optimal Gamma Correction Parameter Estimation)
    # 영역 분리 (임계값 0.5)
    dark_mask = Y_L <= 0.5
    bright_mask = Y_L > 0.5

    s_d = Y_L[dark_mask]
    s_b = Y_L[bright_mask]

    # 어두운 영역 파라미터 추정 (0 < gamma_d <= 1)
    if len(s_d) > 0:
        sigma_d = np.std(s_d)
        gamma_d = 1.0
        
        # Newton's Method를 이용한 반복 추정 (논문의 수식 17 참조)
        for _ in range(100):
            prev_gamma = gamma_d
            val = s_d ** gamma_d
            # 분모 계산 시 로그에 의한 에러 방지용 상수 추가
            log_sd = np.log(s_d + 1e-8)
            
            numerator = np.mean(val) - sigma_d
            denominator = np.mean(val * log_sd)
            
            if denominator == 0:
                break
                
            gamma_d = gamma_d - (numerator / denominator)
            

            # 수렴 조건 (논문 기준 10^-7)
            if abs(gamma_d - prev_gamma) < 1e-7:
                break
        gamma_d = np.clip(gamma_d, 1e-6, 1.0)
    else:
        sigma_d = 0
        gamma_d = 1.0

    # 밝은 영역 파라미터 추정 (1 < gamma_b <= 10)
    if len(s_b) > 0:
        sigma_b = 1.0 - sigma_d
        gamma_b = 1.0
        
        for _ in range(100):
            prev_gamma = gamma_b
            val = s_b ** gamma_b
            log_sb = np.log(s_b + 1e-8)
            
            numerator = np.mean(val) - sigma_b
            denominator = np.mean(val * log_sb)
            
            if denominator == 0:
                break
                
            gamma_b = gamma_b - (numerator / denominator)

            
            if abs(gamma_b - prev_gamma) < 1e-7:
                break
        gamma_b = np.clip(gamma_b, 1.0, 10.0)
    else:
        gamma_b = 1.0

    print("dark side gamma:", gamma_d)
    print("bright side gamma:", gamma_b)
    # 4. 보정된 이미지 융합 (Fusion of Corrected Images)
    Y_d_enhanced = Y_L ** gamma_d
    Y_b_enhanced = Y_L ** gamma_b

    sigma_w = 0.5
    # 가중치 맵 계산 (논문의 수식 19 참조)
    w = np.exp(-(Y_L ** 2) / (2 * (sigma_w ** 2)))

    Y_o = w * Y_d_enhanced + (1.0 - w) * Y_b_enhanced

    # 5. 적응형 색상 복원 (Adaptive Color Restoration)
    # 채도 조절 파라미터 계산 (논문의 수식 20 참조)
    s = 1.0 - np.tanh(Y_L)
    
    # 휘도 비율을 반영한 각 색상 채널 복원
    out_B = Y_o * ((B / (Y_i + 1e-6)) ** s)
    out_G = Y_o * ((G / (Y_i + 1e-6)) ** s)
    out_R = Y_o * ((R / (Y_i + 1e-6)) ** s)

    out_img = cv2.merge([out_B, out_G, out_R])
    out_img = np.clip(out_img, 0, 1) * 255.0
    
    return out_img.astype(np.uint8)

output_image = enhance_low_light_image('img6.jpg')
cv2.imshow('Enhanced Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()