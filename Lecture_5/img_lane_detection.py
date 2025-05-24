import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

from utils import (
    grayscale,
    canny,
    gaussian_blur,
    region_of_interest,
    draw_lines,
    hough_lines,
    weighted_img,
)

# 예시 이미지 불러오기
image = mpimg.imread('test_images/solidWhiteRight.jpg')
# 예시 이미지 정보 출력
print('This image is:', type(image), 'with dimensions:', image.shape)

def average_lines(lines):
    #lines: x1, y1, x2, y2
    
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue # ignore a vertical line
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0: # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    # add more weight to longer lines    
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    
    return left_lane, right_lane # (slope, intercept), (slope, intercept)


def line2pixels(y1, y2, line):
    #Convert a line represented in slope and intercept into pixel points
    
    if line is None:
        return None
    
    slope, intercept = line
    
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    left_lane, right_lane = average_lines(lines)
    
    y1 = image.shape[0] # bottom of the image
    y2 = y1*0.6         # slightly lower than the middle

    left_line  = line2pixels(y1, y2, left_lane)
    right_line = line2pixels(y1, y2, right_lane)
    
    return left_line, right_line

    
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=10):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)  


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
    # Read in and grayscale the image
    image0 = mpimg.imread("test_images/solidWhiteCurve.jpg")
    image=np.copy(image0)
    gray = grayscale(image)

    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    #vertices = np.array([[(50,imshape[0]),(480, 315), (490,315), (imshape[1]-50,imshape[0])]], dtype=np.int32)
    vertices = np.array([[(imshape[1]*0.1,imshape[0]*0.9),(imshape[1]*0.4, imshape[0]*0.65), (imshape[1]*0.6,imshape[0]*0.65), (imshape[1]*0.9,imshape[0]*0.9)]], dtype=np.int32)

    masked_edges=region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    line_img = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    lines, line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    line_image=weighted_img(line_img, image, a=0.8, β=1., λ=0.)


    avg_lane_image = []
    avg_lane_image=draw_lane_lines(image, lane_lines(image, lines))

    plot_images(image0, gray, edges, masked_edges, line_image, avg_lane_image)