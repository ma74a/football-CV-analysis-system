# from ultralytics import YOLO

# model = YOLO("models/best.pt")

# results = model.predict("input_videos/", save=True)
# print(results)
# print("##########################################################################")
# for box in results[0].boxes:
#     print(box)



# """
# cropped player
# for track_id, player_dict in tracks["players"][0].items():
#         bbox = player_dict["bbox"]
#         x1, y1, x2, y2 = map(int, bbox)
#         frame = video_frames[0]
#         cropped_player = frame[y1:y2, x1:x2]   # numpy slicing → [H, W, C]

#         cv2.imwrite("output_videos/cropped_player.jpg", cropped_player)

#         break
#     return
# """

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import read_video

video_frames = list(read_video("input_videos/08fd33_4.mp4"))

first_frame = video_frames[0]

gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
mask_features = np.zeros_like(gray_frame)
mask_features[:,0:20] = 1
mask_features[:,900:1050] = 1
features = cv2.goodFeaturesToTrack(
    gray_frame,
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7,
    mask=mask_features
)
print(gray_frame.shape)
print(mask_features.shape)

for f in features:
    x = int(f[0][0])
    y = int(f[0][1])
    cv2.circle(gray_frame, (x, y), 10, (255, 0, 0), -1)

plt.imshow(first_frame)


# plt.imshow(mask_features)
plt.axis("off")
plt.show()
# gray_frame = cv2.resize(gray_frame, (780, 540), interpolation=cv2.INTER_LINEAR)
# cv2.imshow("img", gray_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()