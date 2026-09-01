from ultralytics import YOLO
import supervision as sv
import os
import pickle

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