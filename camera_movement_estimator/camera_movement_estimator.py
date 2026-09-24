import cv2
import numpy as np
import pickle
import os

from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator:
    def __init__(self, frame):
        # min pixel to see if the camera is moving or not
        self.minimum_distance = 5

        # Lucas-Kanade Optical Flow params
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        )

        old_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # create a masked with the shape of old_frame_gray (w, h)
        mask_features = np.zeros_like(old_frame_gray)
        # mask_features[:,0:20] = 1 # only the top
        # mask_features[:,900:1050] = 1 # only the bottom
        mask_features[0:20, :] = 1
        mask_features[900:1050, :] = 1
        
        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7,
            mask=mask_features
        )

    def get_camera_movement(
        self,
        frames,
        read_from_stubs=False,
        stub_path=None
    ):
        """
        This function will store the camera movement for every frame to [x, y]
        horizontally(x) and vertically(y)
        """

        if read_from_stubs and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
            
        # create a [[0, 0], [0, 0], ...] to store x, y for each frame
        camera_movements = [[0, 0] for _ in range(len(frames))]

        # convert first frame to gray to get the features
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        # get the features that will help us
        """
        [
            [[x1, y1]],
            [[x2, y2]],
            [[x3, y3]],
            ...
        ]
        """
        old_features = cv2.goodFeaturesToTrack(
            old_gray,
            **self.features # unpack this dict
        )

        # loop through all frames and get the features and check if it moves or not
        for frame_num in range(1, len(frames)):
            new_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            # get the new features
            # it gives us how much it moves
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray,
                new_gray,
                old_features,
                None,
                **self.lk_params
            )

            # variable to hold the distance between old and new
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0
            for i, (old, new) in enumerate(zip(old_features, new_features)):
                # get the points of x, y for old, new
                old_features_point = old.ravel()
                new_features_point = new.ravel()

                # get the distance between these values
                distance = measure_distance(
                    pt1=old_features_point,
                    pt2=new_features_point
                )

                # check if there is any distance
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(
                        pt1=old_features_point,
                        pt2=new_features_point
                    )

            # check if the camera is moving or not
            if max_distance > self.minimum_distance:
                camera_movements[frame_num] = [camera_movement_x, camera_movement_y]
                # old features will be to the current frame
                old_features = cv2.goodFeaturesToTrack(new_gray, **self.features)
            # if the camera doesn't move
            else:
                old_features = new_features

            old_gray = new_gray.copy()

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movements, f)

        return camera_movements



    def draw_camera_movement_single_frame(self, frame, movement):
        frame = frame.copy()
        x_movement, y_movement = movement

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        return frame