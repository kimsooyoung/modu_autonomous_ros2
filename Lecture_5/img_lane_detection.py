import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

from utils import (
    grayscale,         # 그레이스케일 변환 함수
    canny,             # 캐니 엣지 검출 함수
    gaussian_blur,     # 가우시안 블러 함수
    region_of_interest,# 관심 영역 마스킹 함수
    draw_lines,        # 선 그리기 함수
    hough_lines,       # 허프 변환으로 직선 검출 함수
    weighted_img,      # 원본 이미지와 선 이미지를 합성하는 함수
)

# 예시 이미지 불러오기
image = mpimg.imread('test_images/solidWhiteRight.jpg')
# 이미지 타입 및 크기 출력
print('이미지 타입:', type(image), '크기:', image.shape)

# 선분들의 평균 기울기와 절편을 계산하여 좌우 차선을 평균화
def average_lines(lines):
    # lines: [[x1, y1, x2, y2], ...] 형태의 선분 리스트

    left_lines    = []  # 왼쪽 차선에 해당하는 (기울기, 절편) 저장
    left_weights  = []  # 해당 선분의 길이를 가중치로 저장
    right_lines   = []  # 오른쪽 차선에 해당하는 (기울기, 절편) 저장
    right_weights = []  # 해당 선분의 길이를 가중치로 저장

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue  # 수직선은 기울기 계산 불가이므로 무시
            slope = (y2 - y1) / (x2 - x1)  # 기울기 계산
            intercept = y1 - slope * x1    # y절편 계산
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)  # 선분 길이 계산
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)

    # 길이를 가중치로 사용하여 평균 기울기와 절편 계산
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane  # (기울기, 절편) 튜플 반환

# (기울기, 절편) 형태의 직선을 실제 이미지 좌표로 변환
def line2pixels(y1, y2, line):
    # line: (기울기, 절편)
    if line is None:
        return None

    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))  # 두 점의 좌표 튜플 반환

# 입력 선분 리스트로부터 좌우 차선을 검출하고 픽셀 좌표로 반환
def lane_lines(image, lines):
    left_lane, right_lane = average_lines(lines)

    y1 = image.shape[0]         # 이미지 맨 아래 (y최대)
    y2 = y1 * 0.6               # 중간보다 약간 아래

    left_line = line2pixels(y1, y2, left_lane)
    right_line = line2pixels(y1, y2, right_lane)

    return left_line, right_line

# 검출된 차선을 이미지에 그려서 반환
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=10):
    # 빈 이미지 생성 후 선 그리기
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    # 원본 이미지와 선 이미지를 합성
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

# 중간 결과들을 시각적으로 비교할 수 있도록 한 번에 출력
def plot_images(original_img, gray_img, edge_img, masked_edge_img, line_img, avg_lane_img):
    # Show all in one figure
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs[0, 0].imshow(original_img)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(gray_img, cmap='gray')
    axs[0, 1].set_title("Grayscale")
    axs[0, 1].axis('off')

    axs[0, 2].imshow(edge_img, cmap='gray')
    axs[0, 2].set_title("Canny Edges")
    axs[0, 2].axis('off')

    axs[1, 0].imshow(masked_edge_img, cmap='gray')
    axs[1, 0].set_title("Masked Edges")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(line_img)
    axs[1, 1].set_title("Hough Lines")
    axs[1, 1].axis('off')

    axs[1, 2].imshow(avg_lane_img)
    axs[1, 2].set_title("Average Lane Lines")
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    # 이미지 읽기 및 복사
    image0 = mpimg.imread("test_images/solidWhiteCurve.jpg")
    image = np.copy(image0)

    # 그레이스케일 변환
    gray = grayscale(image)

    # 가우시안 블러 적용
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # 캐니 엣지 검출
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # 관심 영역 설정 (사다리꼴 형태)
    imshape = image.shape
    vertices = np.array([[
        (imshape[1]*0.1, imshape[0]*0.9),
        (imshape[1]*0.4, imshape[0]*0.65),
        (imshape[1]*0.6, imshape[0]*0.65),
        (imshape[1]*0.9, imshape[0]*0.9)
    ]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # 허프 변환 파라미터 설정 및 적용
    rho = 1               # 거리 해상도
    theta = np.pi / 180   # 각도 해상도 (라디안)
    threshold = 20        # 최소 투표 수
    min_line_length = 20  # 최소 직선 길이
    max_line_gap = 20     # 최대 허용 간격

    line_img = np.copy(image) * 0
    lines, line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # 선 이미지와 원본 이미지 합성
    line_image = weighted_img(line_img, image, a=0.8, β=1., λ=0.)

    # 평균 차선 검출 및 시각화
    avg_lane_image = draw_lane_lines(image, lane_lines(image, lines))

    # 전체 결과 출력
    plot_images(image0, gray, edges, masked_edges, line_image, avg_lane_image)
