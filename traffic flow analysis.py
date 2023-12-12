import cv2
import torch
import numpy as np
import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class YOLODetector():
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self, model_name):
        if model_name == "yolovis":
            model = YOLO(r"E:\Machine learning and deep learning\Projects\object tracking\visdrone yolov8n.pt")
        else:
            model = YOLO('yolov8m.pt')
        return model

    def score_frame(self, frame):
        output_frame = self.model(frame)
        results = output_frame[0]
        return results

class VideoProcessor():
    def __init__(self, videopath, output_videopath):
        self.cap = cv2.VideoCapture(videopath)
        self.output_videopath = output_videopath
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(output_videopath, self.fourcc, self.cap.get(cv2.CAP_PROP_FPS), (int(self.cap.get(3)), int(self.cap.get(4))))
        self.vehicles = []

    def process_video(self, detector, tracker):
        while self.cap.isOpened():
            ret, raw_frame = self.cap.read()

            if not ret:
                break 
            results = detector.score_frame(raw_frame)

            detections = []
            class_labels = results.names
            
            for result in results:
                boxes = result.boxes.data.tolist()
                for i in boxes:
                    x_min, y_min, x_max, y_max, confidence, class_index = i
                    class_index = int(class_index)
                    if True:
                        detections.append([[int(x_min), int(y_min), int(x_max), int(y_max)], confidence, class_index])

            tracks = tracker.update_tracks(detections, frame=raw_frame)

            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = list(track.to_ltwh(orig=True))
                track_id = int(track.track_id)
                
                color_seed = hash(track_id) % (2**32 - 1)
                color = tuple(map(int, np.random.RandomState(color_seed).rand(3) * 255))

                class_id = track.det_class
                classnames = class_labels[class_id]       
                
                cx, cy = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
                
                #cy1, cy2 = 322, 270
                polygon1 = np.array([[(410,1758), (3768,1817), (3772,1971), (304,1914)]], dtype=np.int32)

                # Draw the polygon on the frame
                cv2.polylines(raw_frame, [polygon1], isClosed=True, color=(255, 255, 255), thickness=1)



                if cv2.pointPolygonTest(polygon1, (cx, cy), False) >= 0 and track_id not in self.vehicles and classnames.lower() not in ["people", "pedestrian", "person"]:
                    self.vehicles.append(track_id)
                    print(self.vehicles)

                cv2.rectangle(raw_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(raw_frame, f'{track_id}: {classnames}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(raw_frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(raw_frame, f'Total Vehicles: {len(self.vehicles)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)    
            self.writer.write(raw_frame)
            

        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_name = None  
    detector = YOLODetector("yolovis")
    
    tracker = DeepSort(
        max_iou_distance=0.6,
        max_age=20,
        n_init=2,
        nms_max_overlap=1.0,
        max_cosine_distance=0.5,
        embedder="mobilenet",
        half=True,
        bgr=False,
        embedder_gpu=True
    )

    videopath = r"E:\results of tracking\pexels_videos_2103099 (2160p).mp4"
    output_videopath = r"E:\results of tracking\pexels_videos_2103095 mainn .mp4"

    processor = VideoProcessor(videopath, output_videopath)
    processor.process_video(detector, tracker)