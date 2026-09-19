import cv2

from utils import (
    read_video,
    save_video
)
from tracker import Tracker
from team_assigner import TeamAssign
from player_ball_assigner import PlayerBallAssigner

def main():
    video_frames = list(read_video("input_videos/08fd33_4.mp4"))
    cap = cv2.VideoCapture("input_videos/08fd33_4.mp4")
    tracker = Tracker(model_path="models/best.pt")

    # get tracks from stub or detect
    tracks = tracker.get_object_tracks(
        frames=read_video("input_videos/08fd33_4.mp4"),
        read_from_stub=True,
        stub_path="stubs/track_stubs.pkl"
    )
    tracks["ball"] = tracker.interpolate_ball_detection(tracks["ball"])
    """
    tracks["players"][frame_num][track_id] = {
        "bbox": [x1, y1, x2, y2],
        "team": 1,                      # ← added
        "team_color": [255, 255, 255]   # ← added
    }
    """

    playerball_assigner = PlayerBallAssigner()

    for frame_num, player in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = playerball_assigner.assign_ball_to_player(
            player_dict=player,
            ball_bbox=ball_bbox
        )
        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True

    team_assigner = TeamAssign()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_dict in enumerate(tracks["players"]):
        for track_id, player_bbox in player_dict.items():
            team_id = team_assigner.get_player_team(
                frame=video_frames[frame_num],
                player_bbox=player_bbox["bbox"],
                player_id=track_id
            )

            tracks["players"][frame_num][track_id]["team"] = team_id
            tracks["players"][frame_num][track_id]["team_color"] = team_assigner.team_colors[team_id]


    cap2 = cv2.VideoCapture("input_videos/08fd33_4.mp4")
    ret, first = cap2.read()
    h, w = first.shape[:2]
    out = cv2.VideoWriter(
        "output_videos/output.avi",
        cv2.VideoWriter_fourcc(*'XVID'),
        24, (w, h)
    )

    # process and write one frame at a time
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_num = 0
    while True:
        ret, frame = cap2.read()
        if not ret:
            break

        frame = tracker.draw_annotations(frame, tracks, frame_num)
        out.write(frame)   # write immediately, don't store
        frame_num += 1

    cap2.release()
    out.release()





if __name__ == "__main__":
    main()