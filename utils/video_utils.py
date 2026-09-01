import cv2
import numpy as np
from typing import List

def read_video(video_path: str) -> List[np.ndarray]: # so the array from cv2 is numpy array
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release() 

    return frames


"""
output_video_frames[0].shape[1], output_video_frames[0].shape[0])
shape[1] is for x because x is width
shape[0] is for y because y is height
"""
def save_video(output_video_frames: List[np.ndarray],
               output_video_path: str) -> None:
    if not output_video_frames:
        raise ValueError("No frames to write.")
    
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(filename=output_video_path,
                          fourcc=fourcc,fps=24, 
                          frameSize=(output_video_frames[0].shape[1],   # width
                                      output_video_frames[0].shape[0])) # height
    for frame in output_video_frames:
        out.write(frame)
    out.release()
