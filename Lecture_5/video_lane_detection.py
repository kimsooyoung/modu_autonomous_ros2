from moviepy import *
from moviepy.editor import VideoFileClip

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# 유틸 함수 모듈에서 영상 처리에 필요한 함수들 import
from utils import (
    grayscale,         # 그레이스케일 변환
    canny,             # 캐니 엣지 검출
    gaussian_blur,     # 가우시안 블러
    region_of_interest,# 관심 영역 마스킹
    draw_lines,        # 선 그리기 (테스트용)
    hough_lines,       # 허프 변환을 통한 직선 검출
    weighted_img       # 선 이미지와 원본 이미지 합성
)

# 차선 평균 계산 및 시각화 함수 import
from img_lane_detection import (
    draw_lane_lines,   # 차선을 이미지에 그림
    lane_lines         # 좌우 차선을 평균화하여 계산
)

# 단일 이미지(프레임)에서 차선을 검출하는 파이프라인 함수
def process_image(image):
    # 이미지 -> 차선 검출 결과 이미지 반환

    # 1. 그레이스케일 변환
    gray = grayscale(image)

    # 2. 가우시안 블러 적용
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # 3. 캐니 엣지 검출
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # 4. 관심 영역 정의 및 마스킹 (사다리꼴 형태)
    imshape = image.shape
    vertices = np.array([[
        (imshape[1]*0.1, imshape[0]*0.9),
        (imshape[1]*0.4, imshape[0]*0.65),
        (imshape[1]*0.6, imshape[0]*0.65),
        (imshape[1]*0.9, imshape[0]*0.9)
    ]], dtype=np.int32)

    masked_edges = region_of_interest(edges, vertices)

    # 5. 허프 변환을 위한 파라미터 정의
    rho = 1               # 거리 해상도 (픽셀 단위)
    theta = np.pi / 180   # 각도 해상도 (라디안 단위)
    threshold = 20        # 최소 투표 수
    min_line_length = 20  # 최소 선 길이
    max_line_gap = 20     # 선분 간 최대 간격
    line_img = np.copy(image) * 0  # 선을 그릴 빈 이미지

    # 6. 허프 변환으로 선 검출
    lines, line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # 7. 좌우 차선을 평균화하고 그리기
    avg_lane_image = draw_lane_lines(image, lane_lines(line_img, lines))

    return avg_lane_image  # 최종 결과 이미지 반환

# 영상 파일에 파이프라인 적용
if __name__ == "__main__":
    # 첫 번째 영상 처리: solidWhiteRight.mp4
    white_output = 'test_videos_output/solidWhiteRight.mp4'
    clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image)  # 각 프레임마다 process_image 적용
    white_clip.write_videofile(white_output, audio=False)

    # 두 번째 영상 처리: solidYellowLeft.mp4
    yellow_output = 'test_videos_output/solidYellowLeft.mp4'
    clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
    yellow_clip = clip2.fl_image(process_image)
    yellow_clip.write_videofile(yellow_output, audio=False)
