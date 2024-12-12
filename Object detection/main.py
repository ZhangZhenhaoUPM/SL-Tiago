#Use computer camera to detect objects for test;
#Press 'q' to quit the window;
from ultralytics import YOLO, ASSETS
import cv2

# Load the class name
with open('Object detection\class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

#Load the model
model = YOLO("yolo11n.pt")



#Open the camera
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    #Object detection
    results = model(frame)
    
    #Plot the result
    for result in results:
        for box in result.boxes:

            #Transfer box.xyxy to a numpy array and flatten
            coords = box.xyxy.cpu().numpy().flatten()
            x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            #Got the class name and tranfer it to string
            #label = str(box.cls.item())
            class_id = int(box.cls.item())
            label = class_names[class_id] if class_id < len(class_names) else 'Unknown'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # display performance information
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(frame, f'FPS: {fps}', (0, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow('YOLOv11 Detection', frame)# display Videos
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#for result in results:
    #print(result.boxes.data)
   #result.show()  # uncomment to view each result image
    
    # reference https://docs.ultralytics.com/modes/predict/ for more information.