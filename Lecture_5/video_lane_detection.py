# from moviepy.editor import VideoFileClip
from moviepy import *
from moviepy.editor import VideoFileClip
# from moviepy.video.fx.all import apply_to_frame

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

from img_lane_detection import (
    draw_lane_lines,
    lane_lines
)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    
    # Read in and grayscale the image
    gray = grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
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
    
    avg_lane_image = []
    avg_lane_image=draw_lane_lines(image, lane_lines(line_img, lines))

    return avg_lane_image

white_output = 'test_videos_output/solidWhiteRight_output.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,1)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
# white_clip = clip1.fx(video.fx.all.apply_to_frame, process_image)
# white_clip.write_videofile('test_videos_output/solidWhiteRight_output.mp4', audio=False)

white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)



yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)

print("Done")