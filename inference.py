from ultralytics import YOLO

# Load your trained model
model = YOLO(r"C:\Users\Gunajothi\runs\detect\train4\weights\best.pt")

# Run prediction on test images
results = model.predict(
    source=r"C:\Users\Gunajothi\OneDrive\Desktop\Fruit_Ripeness\data\test\images",
    save=True,
    project=r"C:\Users\Gunajothi\runs\detect",
    name="predict_final",
    exist_ok=True
)
