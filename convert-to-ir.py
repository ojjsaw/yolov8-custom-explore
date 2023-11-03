from ultralytics import YOLO

det_model = YOLO("models/original.pt")

# generates onnx and then auto-converts to openvino IR
# https://docs.ultralytics.com/modes/export/#arguments
det_model.export(format="openvino", imgsz=(416,416), half=False)