# FastSAM-with-video-on-NVIDIA-Jetson

This is a demo for deploying FastSAM to NVIDIA Jetson, and video can be used as input in this demo.

This has been tested and deployed on a [reComputer Jetson J4011](https://www.seeedstudio.com/reComputer-J4011-p-5585.html?queryID=7e0c2522ee08fd79748dfc07645fdd96&objectID=5585&indexName=bazaar_retailer_products). However, you can use any NVIDIA Jetson device to deploy this demo.


## Installation

- **Step 1:** Flash JetPack OS to reComputer Jetson device [(Refer to here)](https://wiki.seeedstudio.com/reComputer_J4012_Flash_Jetpack/).

- **Step 2:** Access the terminal of Jetson device, install pip and upgrade it

```sh
sudo apt update
sudo apt install -y python3-pip
pip3 install --upgrade pip
```

- **Step 3:** Clone the following repo

```sh
git clone https://github.com/CASIA-IVA-Lab/FastSAM
```

- **Step 4:** Open requirements.txt

```sh
cd FastSAM
vi requirements.txt
```

- **Step 5:** Edit the following lines. Here you need to press i first to enter editing mode. Press ESC, then type :wq to save and quit

```sh
# torch>=1.7.0
# torchvision>=0.8.1
```

**Note:** torch and torchvision are excluded for now because they will be installed later.

- **Step 6:** Install the necessary packages

```sh
pip install -r requirements.txt
```

- **Step 7:** Install CLIP

```sh
pip install git+https://github.com/openai/CLIP.git
```

- **Step 8:** Install PyTorch and Torchvision [(Refer to here)](https://wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/#install-pytorch-and-torchvision).

- **Step 9:** Clone this demo.

```sh
git clone https://github.com/yuyoujiang/FastSAM-with-video-on-NVIDIA-Jetson.git
```


## Prepare The Model File

[The pretrained models](https://github.com/CASIA-IVA-Lab/FastSAM#model-checkpoints) are PyTorch models and you can directly use them for inferencing on the Jetson device. However, to have a better speed, you can convert the PyTorch models to TensorRT optimized models by following below instructions.

- **Step 1:** Download model weights in PyTorch format [(Refer to here)](https://github.com/CASIA-IVA-Lab/FastSAM#model-checkpoints).

- **Step 2:** Create a new Python script and enter the following code. Save and execute the file.

```sh
from ultralytics import YOLO


model = YOLO('FastSAM-s.pt')  # load a custom trained
# TensorRT FP32 export
# model.export(format='engine', device='0', imgsz=640)
# TensorRT FP16 export
model.export(format='engine', device='0', imgsz=640, half=True)

```

**Tip:** [Click here](https://docs.ultralytics.com/modes/export) to learn more about yolo export 


## Let's Run It!

### For video 

```sh
python3 Inference_video.py --model_path <path to model> --img_path <path to input video> --imgsz 640
```

### For webcam

```sh
python3 Inference_video.py --model_path <path to model> --img_path <id of camera> --imgsz 640
```

## References

[https://github.com/ultralytics/](https://github.com/ultralytics/)  
[https://github.com/CASIA-IVA-Lab/FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
[https://wiki.seeedstudio.com](https://wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/)


