from ultralytics import YOLO
import supervision as sv
import cv2
import os
import pickle
import numpy as np

from utils import (
    get_bbox_width,
    get_bbox_conter
)

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model=model_path)
        self.tracker = sv.ByteTrack()

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
                color=(255, 255, 255),               # white text on colored rect
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
        for track_id, bboxs in player_dict.items():
            frame = self.draw_ellipse(frame, bboxs["bbox"], [255, 0, 0], track_id)

        # draw referee
        for _, bboxs in referee_dict.items():
                    frame = self.draw_ellipse(frame, bboxs["bbox"], [0, 225, 255])

        # draw ball triangle
        for _, bbox in ball_dict.items():
            frame = self.draw_triangle(frame, bbox["bbox"], [0, 255, 0])
            

        return frame
