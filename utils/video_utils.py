import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame  # one frame at a time, not all at once
    cap.release()


def save_video(frames, output_path="output_videos/"):
    if not frames:
        return "Not frames"
    video_path = os.path.join(output_path, "output.avi")
    h, w = frames[0].shape[0], frames[0].shape[1]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        filename=video_path,
        fourcc=fourcc,
        fps=24,
        frameSize=(w, h) #  # OpenCV wants (width, height)
    )
    for frame in frames:
        out.write(frame)

    out.release()
    print("#######SAVED#######")