#loading libraries
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
from violation_clasess import VIOLATION_CLASSES

# PATHS
MODEL_PATH = r"D:\Final year project traffic violation detection\weights\best (1).pt"
IMAGE_PATH = r"D:\Final year project traffic violation detection\images\valid\images\-_-_20230830_105407_0065_jpg.rf.e0c1e7535c0fe668ea1d6ca4c9e03786.jpg"
OUTPUT_DIR = r"D:\Final year project traffic violation detection\outputs\results"

os.makedirs(OUTPUT_DIR, exist_ok=True)
# LOAD MODEL 
model = YOLO(MODEL_PATH)

# RUN INFERENCE 
results = model(IMAGE_PATH, conf=0.4) #model predicts the objects and confidence threshold set to 0.4, predictions below this are ignored

# to read the input image
img = cv2.imread(IMAGE_PATH)

# DRAW RESULTS 
for r in results:# yolo returns multiple objects so loop through each object
    for box in r.boxes:
        cls_id = int(box.cls[0]) # get class id
        conf = float(box.conf[0]) # get confidence score

        label = VIOLATION_CLASSES.get(cls_id, "Unknown") # converts class ids to violations

        # Get Bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Red = violation, Green = normal
        color = (0, 0, 255) if "/" in label else (0, 255, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2) #draws the boudding boxes to objects in image
        cv2.putText(
            img,
            f"{label} ({conf:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        ) # the detected classname and confidence score printed on objects in the image


# SAVE OUTPUT WITH TIMESTAMP 
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_filename = f"violation_{timestamp}.jpg"
output_path = os.path.join(OUTPUT_DIR, output_filename)

cv2.imwrite(output_path, img)
print(f"[INFO] Output saved at: {output_path}")

# SHOW OUTPUT 
cv2.imshow("Traffic Violation Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

