from sklearn.cluster import KMeans

class TeamAssign:
    def __init__(self):
        self.team_colors = {} #  {1: [R,G,B], 2: [R,G,B]}
        self.player_team_dict = {} # {track_id: team_id} — cache per player

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        cropped_player = frame[y1:y2, x1:x2] # [H, W, C]
        # get only the top half of the player
        top_half_player = cropped_player[:int(cropped_player.shape[0]/2):] # [H, W, C]
        # we'll reshape this to [H*W, C] to give to the model
        reshaped_img = top_half_player.reshape(-1, 3)

        kmeans_model = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans_model.fit(reshaped_img)

        # get the cluster labels Which is [0, 1]
        labels = kmeans_model.labels_ # shape=(2156,)
        # let's reshape it to [h, w]
        clustered_image = labels.reshape(top_half_player.shape[0], top_half_player.shape[1])

        # figure out which cluster is background (corners are almost always grass)
        corners = [
            clustered_image[0, 0],   # top-left
            clustered_image[0, -1],  # top-right
            clustered_image[-1, 0],  # bottom-left
            clustered_image[-1, -1]  # bottom-right
        ]
        bg_cluster = max(set(corners), key=corners.count)  # most common corner cluster
        player_cluster = 1 - bg_cluster

        player_color = kmeans_model.cluster_centers_[player_cluster]  # RGB color


        return player_color

    def assign_team_color(self, frame, player_detections):
        """
        get the color of each team players and then assign it for each team [1, 2]
        player_detections -> track_id, bbox
        """
        players_color = [] # list to hold all players color from each team
        for _, player_bbox in player_detections.items():
            bbox = player_bbox["bbox"]
            player_color = self.get_player_color(frame, bbox)
            players_color.append(player_color)

        # then cluster players color for each team
        kmeans_model = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans_model.fit(players_color)

        self.kmeans_model = kmeans_model

        self.team_colors[1] = kmeans_model.cluster_centers_[0] # team 1 color
        self.team_colors[2] = kmeans_model.cluster_centers_[1] # team 2 color

    def get_player_team(self,frame,player_bbox,player_id):
         # cache — same player keeps same team across frames
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)
        # predict which team cluster this color belongs to
        team_id = self.kmeans_model.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        if player_id ==91:
            team_id=1

        self.player_team_dict[player_id] = team_id

        return team_id