from openvino.runtime import Core, Model

core = Core()
det_ov_model = core.read_model("models/original_openvino_model/original.xml")
if device.value != "CPU":
    det_ov_model.reshape({0: [1, 3, 640, 640]})
det_compiled_model = core.compile_model(det_ov_model, device.value)