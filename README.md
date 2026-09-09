## WORKSHOP-2

### Real-Time Object Detection using Web Camera and YOLOv8

Welcome to Workshop 2! This repository demonstrates how to perform real-time object detection using a webcam feed powered by the YOLOv8 deep learning model. It captures live video streams from your local camera, detects objects across video frames, and dynamically renders annotated bounding boxes and labels directly within a Jupyter Notebook environment.

## Features:
- Captures live video feed in real-time using a webcam.
- Performs accurate multi-class object detection using the pre-trained YOLOv8 model (`yolov8m.pt`).
- Draws bounding boxes and class labels with a customized confidence threshold.
- Renders animated frame updates inline using Matplotlib and IPython display clearing.

## Technologies Used:
- Python
- Ultralytics (YOLOv8) for real-time object detection
- OpenCV for video stream capture and color channel conversions
- Matplotlib and IPython for inline frame rendering and display handling

## Program & output : 
**NAME** : THENAMIZHTHAN V

**REG. NO.** : 212225240175

### importing libraries and open-cv - 
```
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
```

### Load a more accurate YOLO model
```
model = YOLO("yolov8m.pt")
```

### Open webcam
```
cap = cv2.VideoCapture(0)
```

### Check if camera opened
```
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame.")
            break

        # Perform object detection
        results = model(
            frame,
            conf=0.60,       
            verbose=False
        )

        # Draw bounding boxes and labels
        annotated_frame = results[0].plot()

        # Convert BGR to RGB for Matplotlib
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display in Jupyter
        clear_output(wait=True)
        plt.figure(figsize=(10, 8))
        plt.imshow(annotated_frame)
        plt.title("Real-Time Object Detection")
        plt.axis("off")
        plt.show()
```

### Release webcam
```
cap.release()
cv2.destroyAllWindows()
```

## ouput-

<img width="942" height="622" alt="image" src="https://github.com/user-attachments/assets/af9601bd-afbc-44aa-b8b7-3637c02ea328" />

## RESULT - 
  Thus , the Real-Time Object Detection using Web Camera and YOLOv8 has been done sucessfully.
