from ultralytics import YOLO

model = YOLO("models/yolov8l.pt")

results = model.predict("input_videos/08fd33_4.mp4", save=True)
print(results[0])
print(len(results))