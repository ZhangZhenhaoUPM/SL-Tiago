from ultralytics import YOLO, ASSETS

model = YOLO("yolo11n.pt", task="detect")
results = model(source=ASSETS / "bus.jpg")

for result in results:
    print(result.boxes.data)
    # result.show()  # uncomment to view each result image
    
    # reference https://docs.ultralytics.com/modes/predict/ for more information.