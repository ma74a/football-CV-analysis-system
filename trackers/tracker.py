from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import os
import pickle

from utils import (
    get_bbox_width,
    get_bbox_center
)

class Tracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model=model_path)
        self.tracker = sv.ByteTrack()

    def get_detections(self, frames):
        """Get the objects detections and return them as a list,
        detect all the frames but in batches not the whole frames at once
        """

        frames_batch_size=20
        detections = []
        for i in range(0, len(frames), frames_batch_size):
            batch_frames = self.model.predict(frames[i:i+frames_batch_size], conf=0.1)
            detections.extend(batch_frames)

        return detections


    def get_objects_tracks(self, frames, read_from_stub, stub_path):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.get_detections(frames)

        """
        for each frame we'll get for each object the track id and bbox
        frame_num = 0 the first index in the list which contains {track_id: bbox}
        "players": [
            {0: {track_id: bbox}, {track_id: bbox}} which 0 for frame_num this for only one frame
            {1: {track_id: bbox}, {track_id: bbox}} and so
        ]
        """
        tracks = {
            "players": [], 
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names # {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
            cls_index_to_name = {v: k for k, v in cls_names.items()}

            # convert to supervision format
            supervision_detections = sv.Detections.from_ultralytics(detection)

            # convert goalkeeper id to player id
            for cls_id_index, cls_id in enumerate(supervision_detections.class_id):
                if cls_names[cls_id] == "goalkeeper":
                    supervision_detections.class_id[cls_id_index] = cls_index_to_name["player"]

            # Track objects
            detections_with_tracks = self.tracker.update_with_detections(supervision_detections)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detections in detections_with_tracks:
                bbox = frame_detections[0].tolist()
                cls_frame_id = frame_detections[3]
                track_id = frame_detections[4]

                if cls_frame_id == cls_index_to_name["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_frame_id == cls_index_to_name["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in supervision_detections:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_index_to_name['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

            if stub_path is not None:
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_bbox_center(bbox)
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
                color=(255, 0, 0),
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
        x_center, _ = get_bbox_center(bbox)
        y1 = bbox[1] # top y

        triangle_points = np.array([
        [x_center - 10, y1 - 15],   # base left  (top)
        [x_center + 10, y1 - 15],   # base right (top)
        [x_center,      y1],        # tip pointing DOWN toward ball ← flip this
        ], dtype=np.int32)

        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2) # like the boarder

        return frame


    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw players ellipse
            for track_id, player in player_dict.items():
                bbox = player["bbox"]
                frame = self.draw_ellipse(frame, bbox, (255, 0, 0), track_id)

            # Draw Referees ellipse
            for _, referee in referee_dict.items():
                bbox = referee["bbox"]
                frame = self.draw_ellipse(frame, bbox, (255, 255, 0))

            # Draw ball triangle
            for _, ball in ball_dict.items():
                bbox = ball["bbox"]
                frame = self.draw_triangle(frame, bbox, (0,255,0))

            output_video_frames.append(frame)

        return output_video_frames