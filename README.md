<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">train_yolo_v9</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/train_yolo_v9">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/train_yolo_v9">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/train_yolo_v9/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/train_yolo_v9.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>


Train on YOLOv9 object detection models.

![London street object detection](https://raw.githubusercontent.com/Ikomia-hub/infer_yolo_v9/main/images/output.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add dataset loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "detection",
}) 

# Add train algorithm
train = wf.add_task(name="train_yolo_v9", auto_connect=True)

# Launch your training on your data
wf.run()
```

## :sunny: Use with Ikomia Studio
Ikomia Studio offers a friendly UI with the same features as the API.
- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters
- **model_name** (str) - default 'yolov9-c': Model architecture to be trained. Should be one of :
    - yolov9-s
    - yolov9-m
    - yolov9-c
    - yolov9-e
- **train_imgsz** (int) - default '640': Size of the training image.
- **test_imgsz** (int) - default '640': Size of the eval image.
- **epochs** (int) - default '50': Number of complete passes through the training dataset.
- **batch_size** (int) - default '8': Number of samples processed before the model is updated.
- **dataset_split_ratio** (float) – default '0.9': Divide the dataset into train and evaluation sets ]0, 1[.
- **output_folder** (str, *optional*): Path to where the model will be saved. 
- **config_file** (str, *optional*): Path to hyperparameters configuration file .yaml. 
- **dataset_folder** (str, *optional*): Path to dataset folder.
- **model_weight_file** (str, *optional*): Path to pretrained model weights. Can be used to fine tune a model.

**Parameters** should be in **strings format**  when added to the dictionary.


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add dataset loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "detection",
}) 

# Add train algorithm
train = wf.add_task(name="train_yolo_v9", auto_connect=True)
train.set_parameters({
    "batch_size": "4",
    "epochs": "5",
    "train_imgsz": "640",
    "test_imgsz": "640",
    "dataset_split_ratio": "0.9"
})
```

