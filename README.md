# yolov8-custom-explore

```sh
pip install openvino-dev[caffe,kaldi,mxnet,onnx,pytorch,tensorflow2] ultralytics nncf

# only for LibGL error in cloud envr. for opencv
sudo apt-get install ffmpeg libsm6 libxext6  -y
```

### Yolov8 Object Detection
* Used RoboFlow for dataset
* Used Google Colab to train - 3 epochs - labels: head, helmet, person
* Convert Yolov8 model to IR
    ```sh
    python convert-to-ir.py

    # outputs while conversion
    # starting from 'models/original.pt' with input shape (1, 3, 416, 416) BCHW and output shape(s) (1, 7, 3549) (21.5 MB)
    ```

* Convert Yolov8 model to IR
    ```sh
    python nncf-explore.py
    ```

* Benchmark FP32 OV IR Model
    ```sh
    # example for benchmark_app for above converted custom yolov8 ov model
    benchmark_app -m "models/original_openvino_model/original.xml" -d CPU -hint throughput -t 15
    # [ INFO ] Throughput:   7.08 FPS
    ```

* Benchmark INT8 OV IR Model
    ```sh
    # example for benchmark_app for int8 nncf quantized ov model
    benchmark_app -m "models/nncf_int8_model/quantized_model.xml" -d CPU -hint throughput -t 15

    # [ INFO ] Throughput:   12.63 FPS
    ```

### Yolov8 Instance Segmentation
### Yolov8 Keypoint Detection
