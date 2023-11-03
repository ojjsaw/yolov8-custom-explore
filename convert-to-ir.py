from ultralytics import YOLO

det_model = YOLO("models/original.pt")

# generates onnx and then auto-converts to openvino IR
det_model.export(format="openvino", dynamic=True, half=False)