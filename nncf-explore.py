import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import nncf
import openvino as ov
from PIL import Image

TEST_IMGS_DIR_PATH="train/images"
OV_FP32_MODEL_XML_PATH="models/original_openvino_model/original.xml"
TARGET_W=416
TARGET_H=416

class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize((TARGET_H, TARGET_W)),  # Resize images
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
])

image_dataset = CustomImageDataset(directory=TEST_IMGS_DIR_PATH, transform=transform)
calibration_loader = torch.utils.data.DataLoader(image_dataset)

def transform_fn(image_data):
    return image_data.numpy()

calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)

# load fp32 ov model and quantize
model = ov.Core().read_model(OV_FP32_MODEL_XML_PATH)
print("Quantizing with the number of images provided. Min. 300 recommended")
quantized_model = nncf.quantize(model, 
                                calibration_dataset,
                                target_device=nncf.TargetDevice.CPU,
                                preset=nncf.QuantizationPreset.MIXED,
                                ignored_scope=nncf.IgnoredScope(
                                        types=["Multiply", "Subtract", "Sigmoid"],  # ignore operations
                                        names=[
                                            "/model.22/dfl/conv/Conv",           # in the post-processing subgraph
                                            "/model.22/Add",
                                            "/model.22/Add_1",
                                            "/model.22/Add_2",
                                        ]
                                    )
                                )

model_int8 = ov.compile_model(quantized_model)

# export the model
path_to_create = Path("models/nncf_int8_model")
path_to_create.mkdir(parents=True, exist_ok=True)
ov.serialize(quantized_model, "models/nncf_int8_model/quantized_model.xml")




# https://medium.com/openvino-toolkit/developers-hands-on-segment-anything-quantitative-acceleration-4d1c31cb07d1