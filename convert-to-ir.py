from ultralytics import YOLO

det_model = YOLO("models/original.pt")

# generates onnx and then auto-converts to openvino IR
# https://docs.ultralytics.com/modes/export/#arguments
# half=True for FP16 but FP32 preferred for int8 quantization
det_model.export(format="openvino", imgsz=(416,416), half=False)