from ultralytics import YOLO

# Load a model
model = YOLO("models/original.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model("test/images/000008_jpg.rf.COFigDuPuObOGpfjmTy4.jpg")  # predict on an image

print(results)