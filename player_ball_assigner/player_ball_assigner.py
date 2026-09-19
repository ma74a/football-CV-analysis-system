from utils import get_bbox_conter, measure_distance

class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, player_dict, ball_bbox):
        ball_center_position = get_bbox_conter(ball_bbox)

        min_distance = float("inf")
        assigned_player = -1
        for player_id, player in player_dict.items():
            player_bbox = player["bbox"]

            """
            (x1,y1) ──────────── (x2,y1)
                |                     |
                |       player        |
                |                     |
                (x1,y2) ──────────── (x2,y2)
                ↑                     ↑
                distance_left        distance_right
                    ↘                 ↙
                        ⚽ ball_position
            """
            left_distance = measure_distance(
                pt1=(player_bbox[0], player_bbox[3]), # (x1, y2)  ← bottom-left  corner
                pt2=ball_center_position 
            )
            right_distance = measure_distance(
                pt1=(player_bbox[2], player_bbox[3]), # (x2, y2)   ← bottom-right  corner
                pt2=ball_center_position 
            )

            distance = min(left_distance, right_distance)
            if distance < self.max_player_ball_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id

        return assigned_player # return the player_id which is track_id