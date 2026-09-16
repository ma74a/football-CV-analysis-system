from ultralytics import YOLO
import supervision as sv
import cv2

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model=model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        detections = []
        batch_size = 20
        # detect with batches of frames not frame by frame
        for i in range(0, len(frames), batch_size):
            batch = frames[i: i+batch_size]
            result = self.model.predict(batch, stream=True)
            detections.extend(result)
            break

        return detections

    def get_object_tracks(self, frames):
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

        return tracks

