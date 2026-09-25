import math

def get_bbox_conter(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2) / 2), int((y1+y2) / 2)

def get_bbox_width(bbox):
    x1, y1, x2, y2 = bbox 
    return int(x2 - x1)

def measure_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def measure_xy_distance(pt1, pt2):
    x_dist = pt1[0] - pt2[0]  # old_x - new_x
    y_dist = pt1[1] - pt2[1]  # old_y - new_y
    return x_dist, y_dist


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2) / 2), int(y2)