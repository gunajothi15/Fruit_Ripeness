from ultralytics import YOLO
import cv2
from tkinter import Tk, filedialog

# Load trained model
model = YOLO(r"C:\Users\Gunajothi\runs\detect\train4\weights\best.pt")

# Class labels (must match your data.yaml order)
class_names = ['raw_banana', 'raw_mango', 'ripe_banana', 'ripe_mango']

def camera_mode():
    cap = cv2.VideoCapture(0)
    print("Press 'c' to capture and predict, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Live Feed - Press 'c' to Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            results = model.predict(source=frame, save=False, conf=0.3, verbose=False)

            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = f"{class_names[cls_id]} {conf:.2f}"

                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        print(f"Detected: {class_names[cls_id]} (Confidence: {conf:.2f})")

            cv2.imshow("Captured Prediction", frame)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def upload_mode():
    Tk().withdraw()  # hide main Tkinter window
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_path:
        print("No file selected.")
        return

    results = model.predict(source=file_path, save=False, conf=0.3, verbose=False)

    img = cv2.imread(file_path)
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{class_names[cls_id]} {conf:.2f}"

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                print(f"Detected: {class_names[cls_id]} (Confidence: {conf:.2f})")

    cv2.imshow("Uploaded Image Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Choose an option:\n1. Camera Mode\n2. Upload Image")
    choice = input("Enter choice (1/2): ")

    if choice == "1":
        camera_mode()
    elif choice == "2":
        upload_mode()
    else:
        print("Invalid choice.")
