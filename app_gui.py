# app_gui.py (Step 3: polished app)
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import os

# Load your trained model
model = YOLO(r"C:\Users\Gunajothi\runs\detect\train4\weights\best.pt")

# Class labels (match your data.yaml)
class_names = ['raw_banana', 'raw_mango', 'ripe_banana', 'ripe_mango']

# ------------------ NUTRITION INFO (MACROS) ------------------
nutrition_info = {
    'raw_banana': {
        'Calories': '89 kcal',
        'Carbs': '22.8 g',
        'Protein': '1.1 g',
        'Fat': '0.3 g',
        'Fiber': '2.6 g',
        'Sugar': '12 g'
    },
    'ripe_banana': {
        'Calories': '89 kcal',
        'Carbs': '23 g',
        'Protein': '1.1 g',
        'Fat': '0.3 g',
        'Fiber': '2.6 g',
        'Sugar': '17 g'
    },
    'raw_mango': {
        'Calories': '60 kcal',
        'Carbs': '15 g',
        'Protein': '0.8 g',
        'Fat': '0.4 g',
        'Fiber': '1.6 g',
        'Sugar': '14 g'
    },
    'ripe_mango': {
        'Calories': '70 kcal',
        'Carbs': '17 g',
        'Protein': '0.8 g',
        'Fat': '0.6 g',
        'Fiber': '2 g',
        'Sugar': '15 g'
    }
}
# --------------------------------------------------------------


def show_nutrition_popup(cls_id):
    """Shows nutrition popup for the detected fruit."""
    fruit_name = class_names[cls_id]
    info = nutrition_info.get(fruit_name)

    if info:
        msg = f"{fruit_name.replace('_', ' ').title()} Nutrition:\n\n"
        for k, v in info.items():
            msg += f"{k}: {v}\n"

        messagebox.showinfo("Nutrition Info", msg)


def draw_predictions(image, results):
    """Draw YOLO predictions on the frame"""
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{class_names[cls_id]} {conf:.2f}"
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Draw box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return image

    """Draw YOLO predictions on the frame"""
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{class_names[cls_id]} {conf:.2f}"
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Draw box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Show nutrition popup
                show_nutrition_popup(cls_id)

    return image


def on_camera_mode():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam")
        return

    messagebox.showinfo("Camera Mode", "Press 'C' to capture, 'Q' to quit webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Live Camera Feed - Press C to Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        # Quit webcam
        if key == ord('q'):
            break

        # Capture frame and show prediction + nutrition
        if key == ord('c'):
            results = model.predict(source=frame, save=False, conf=0.4, verbose=False)
            annotated = draw_predictions(frame.copy(), results)

            # Convert to Tkinter image
            image_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(image_rgb)
            imgtk = ImageTk.PhotoImage(im_pil)

            # New window
            top = tk.Toplevel()
            top.title("Camera Capture Result")

            lbl_img = tk.Label(top, image=imgtk)
            lbl_img.image = imgtk
            lbl_img.pack(pady=10)

            # Extract nutrition details
            fruit_label = "No fruit detected"

            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    cls_id = int(r.boxes[0].cls[0])
                    fruit_name = class_names[cls_id]
                    info = nutrition_info.get(fruit_name)

                    nutri_text = f"Fruit: {fruit_name.replace('_',' ').title()}\n\n"
                    for k, v in info.items():
                        nutri_text += f"{k}: {v}\n"

                    fruit_label = nutri_text
                    break

            lbl_nutrition = tk.Label(top, text=fruit_label, font=("Segoe UI", 12), justify="left")
            lbl_nutrition.pack(pady=10)

    cap.release()
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam")
        return

    messagebox.showinfo("Camera Mode", "Press 'Q' to quit webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO on each frame (real-time)
        results = model.predict(source=frame, save=False, conf=0.4, verbose=False)
        frame = draw_predictions(frame, results)

        cv2.imshow("Live Camera Detection - Press Q to Quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def on_upload_mode():
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    image = cv2.imread(file_path)
    if image is None:
        messagebox.showerror("Error", "Failed to load image")
        return

    results = model.predict(source=image, save=False, conf=0.4, verbose=False)
    annotated = draw_predictions(image.copy(), results)

    # Convert to Tkinter image
    image_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(image_rgb)
    imgtk = ImageTk.PhotoImage(im_pil)

    # Create window
    top = tk.Toplevel()
    top.title("Prediction Result")

    # Show image
    lbl_img = tk.Label(top, image=imgtk)
    lbl_img.image = imgtk
    lbl_img.pack(pady=10)

    # ---- Extract first detected fruit ----
    fruit_label = "No fruit detected"

    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            cls_id = int(r.boxes[0].cls[0])
            fruit_name = class_names[cls_id]
            info = nutrition_info.get(fruit_name)

            # Build nutrition text
            nutri_text = f"Fruit: {fruit_name.replace('_',' ').title()}\n\n"
            for k, v in info.items():
                nutri_text += f"{k}: {v}\n"

            fruit_label = nutri_text
            break

    # Show nutrition info below image
    lbl_nutrition = tk.Label(top, text=fruit_label, font=("Segoe UI", 12), justify="left")
    lbl_nutrition.pack(pady=10)


def build_gui():
    root = tk.Tk()
    root.title("🍌🥭 Fruit Ripeness Scanner")
    root.geometry("420x200")
    root.resizable(False, False)

    # Optional: set app icon (replace 'icon.ico' with your file)
    if os.path.exists("icon.ico"):
        root.iconbitmap("icon.ico")

    header = tk.Label(root, text="Fruit Ripeness Scanner",
                      font=("Segoe UI", 16, "bold"))
    header.pack(pady=(15, 10))

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)

    cam_btn = tk.Button(btn_frame, text="📸 Camera Mode",
                        width=18, height=2, command=on_camera_mode)
    cam_btn.grid(row=0, column=0, padx=15)

    upload_btn = tk.Button(btn_frame, text="📂 Upload Image",
                           width=18, height=2, command=on_upload_mode)
    upload_btn.grid(row=0, column=1, padx=15)

    footer = tk.Label(root, text="Powered by YOLOv8 + Tkinter",
                      font=("Segoe UI", 9), fg="gray")
    footer.pack(side="bottom", pady=8)

    root.mainloop()


if __name__ == "__main__":
    build_gui()
