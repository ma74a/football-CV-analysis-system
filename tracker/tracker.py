from ultralytics import YOLO
import supervision as sv
import cv2
import os
import pickle
import numpy as np
import pandas as pd

from utils import (
    get_bbox_width,
    get_bbox_conter,
    get_foot_position
)

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model=model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        """add position of ball or player or referee to the tracks"""
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    # if the object is ball 
                    # get the center of it
                    if object == "ball":
                        position = get_bbox_conter(bbox)
                    # if otherwise get the foot position
                    else:
                        position = get_foot_position(bbox)

                    tracks[object][frame_num][track_id]["position"] = position

    def interpolate_ball_detection(self, ball_positions):
        """
        # input — ball missing in frames 2,3,4
        ball_positions = [
            {1: {"bbox": [100, 200, 110, 210]}},  # frame 0 ✅
            {1: {"bbox": [120, 205, 130, 215]}},  # frame 1 ✅
            {},                                    # frame 2 ❌ missed
            {},                                    # frame 3 ❌ missed
            {1: {"bbox": [180, 220, 190, 230]}},  # frame 4 ✅
        ]

        # step 1 — extract bboxes
        [
        [100, 200, 110, 210],  # frame 0
        [120, 205, 130, 215],  # frame 1
        [],                    # frame 2 → becomes NaN
        [],                    # frame 3 → becomes NaN
        [180, 220, 190, 230],  # frame 4
        ]
        
        step 2 — after interpolate()
        frame 2 → [140, 210, 150, 220]  ← linearly interpolated
        frame 3 → [160, 215, 170, 225]  ← linearly interpolated
        """
        # ball_positions = list of dicts, one per frame
        # [{1: {"bbox": [...]}}, {}, {1: {"bbox": [...]}}, ...]
        #                         ↑ empty = ball not detected this frame

        # extract bbox or None for each frame
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]

        # convert to dataframe
        df = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        # replace empty lists with NaN so pandas can interpolate
        df = df.replace({None: pd.NA})

        # interpolate missing values linearly
        df = df.interpolate()

        # fill any remaining NaN at start/end (interpolate won't fill edges)
        df = df.bfill().ffill()

        # convert back to original format
        ball_positions = [{1: {"bbox": x}} for x in df.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames, batch_size=20):
        for i in range(0, len(frames), batch_size):
            batch = frames[i: i + batch_size]
            results = self.model.predict(batch, stream=True, verbose=False)
            yield from results  # don't accumulate

    def get_object_tracks(
            self,
            frames,
            read_from_stub=False,
            stub_path=None
        ):
        # check if read from saved stubs or not
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)

        """
        players: frame_num: track_id: bbox
        frame_num is the list index
        list of frame numbers contains track_id that contains dicts of bbox for each player
        across each frame
        # frame 0 → {3: {"bbox": [...]}, 7: {"bbox": [...]}}
        """
        tracks = {
            "players": [] ,
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            """
            detection is only one frame, 
            len of it gives the number of objects in this frame
            # print(len(detection)) 
            """
            # names: {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
            idx_to_names = detection.names
            names_to_idx = {v: k  for k, v in idx_to_names.items()}

            supervision_detection = sv.Detections.from_ultralytics(detection)

            # convert goalkeeper into player 
            for class_index, class_id in enumerate(supervision_detection.class_id):
                if class_id == names_to_idx["goalkeeper"]:
                    supervision_detection.class_id[class_index] = names_to_idx["player"]

            detections_with_tracks = self.tracker.update_with_detections(supervision_detection)

            tracks["players"].append({}) # dict for bbox
            tracks["referees"].append({})
            tracks["ball"].append({})

            for detect_track in detections_with_tracks:
                bbox = detect_track[0].tolist()
                cls_id = detect_track[3]
                track_id = detect_track[4]

                if cls_id == names_to_idx["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == names_to_idx["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in supervision_detection:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == names_to_idx['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(
        self,
        frame,
        bbox,
        color,
        track_id=None
    ):
        y2 = int(bbox[3])
        x_center, _ = get_bbox_conter(bbox)
        bbox_width = get_bbox_width(bbox)

        cv2.ellipse(
            img=frame,
            center=(x_center, y2),
            axes=(int(bbox_width / 2), 8),  # half-width matches player width, small height for flat look
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Draw rectangle + track ID inside the ellipse
        rect_width = 40
        rect_height = 20
        x1_rect = x_center - rect_width // 2
        x2_rect = x_center + rect_width // 2
        y1_rect = y2 - rect_height // 2 + 15
        y2_rect = y2 + rect_height // 2 + 15

        if track_id is not None:
            cv2.rectangle(
                img=frame,
                pt1=(x1_rect, y1_rect),
                pt2=(x2_rect, y2_rect),
                color=color,
                thickness=cv2.FILLED
            )
            # x1_text = x1_rect+12
            # if track_id > 99:
            #     x1_text -=10

            cv2.putText(
                img=frame,
                text=str(track_id),
                org=(x1_rect + 8, y2_rect - 5),      # slight padding inside rect
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 0),               # white text on colored rect
                thickness=2
            )

        return frame

    """
    [x_center - 10, y1 - 15]●-----------● [x_center + 10, y1 - 15]
                             \         /
                              \       /
                               \     /
                                \   /
                                 \ /
                                  ● [x_center, y1]   ← tip
    """
    def draw_triangle(self, frame, bbox, color):
        x_center, _ = get_bbox_conter(bbox)
        y1 = bbox[1] # top y

        triangle_points = np.array([
        [x_center - 10, y1 - 15],   # base left  (top)
        [x_center + 10, y1 - 15],   # base right (top)
        [x_center,      y1],        # tip pointing DOWN toward ball ← flip this
        ], dtype=np.int32)

        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2) # like the boarder

        return frame

    def draw_annotations(self, frame, tracks, frame_num):
        frame = frame.copy()
        player_dict = tracks["players"][frame_num]
        referee_dict = tracks["referees"][frame_num]
        ball_dict = tracks["ball"][frame_num]

        # draw player
        for track_id, player in player_dict.items():
            color = player.get("team_color", [0, 0, 255])
            frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
            if player.get("has_ball", False):
                frame = self.draw_triangle(frame, player["bbox"], [0, 0, 255])

        # draw referee
        for _, bboxs in referee_dict.items():
                    frame = self.draw_ellipse(frame, bboxs["bbox"], [0, 225, 255])

        # draw ball triangle
        for _, bbox in ball_dict.items():
            frame = self.draw_triangle(frame, bbox["bbox"], [0, 255, 0])
            

        return frame
