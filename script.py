from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO(r"C:\Users\Gunajothi\runs\detect\train4\weights\best.pt")

# Class labels (must match your data.yaml order)
class_names = ['raw_banana', 'raw_mango', 'ripe_banana', 'ripe_mango']

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'c' to capture and predict, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show live video feed
    cv2.imshow("Live Feed - Press 'c' to Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # Capture frame
        # Run YOLO on captured frame
        results = model.predict(source=frame, save=False, conf=0.3, verbose=False)  # lowered conf

        # Loop through detections
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f"{class_names[cls_id]} {conf:.2f}"

                    # Get coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # Draw rectangle and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    print(f"Detected: {class_names[cls_id]} (Confidence: {conf:.2f})")

        # Show captured + detected image
        cv2.imshow("Captured Prediction", frame)

    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
