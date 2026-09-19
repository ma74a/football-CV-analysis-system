import math

def get_bbox_conter(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2) / 2), int((y1+y2) / 2)

def get_bbox_width(bbox):
    x1, y1, x2, y2 = bbox 
    return int(x2 - x1)

def measure_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)