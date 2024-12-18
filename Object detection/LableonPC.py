import cv2
import torch
from ultralytics import YOLO, ASSETS
import cv2
import os
import numpy

#Input and Output Folders
input_folder = 'D:\githubProject\SL-Tiago\Object detection\input\RGB——robotathome'
output_folder = 'D:\githubProject\SL-Tiago\Object detection\output'
os.makedirs(output_folder, exist_ok=True)

#model = YOLO("yolo11m-seg.yaml")  # build a new model from YAML
model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo11n-seg.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

allowed_classes = ['bed', 'sofa', 'picture_frame']

#image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

for img_name in os.listdir(input_folder):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img_path = os.path.join(input_folder, img_name)
        frame = cv2.imread(img_path)
        results = model(frame)
  
        
    for result in results:
        for box in result.boxes:

            #Transfer box.xyxy to a numpy array and flatten
            coords = box.xyxy.cpu().numpy().flatten()
            x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            #Got the class name and tranfer it to string
            class_id = int(box.cls.item())
            label = model.names[class_id] if class_id < len(model.names) else 'Unknown'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        
        # save image
        save_path = os.path.join(output_folder, img_name)
        cv2.imwrite(save_path, frame)
        print(f"Processed and saved {save_path}")
        #model.export(format="onnx")