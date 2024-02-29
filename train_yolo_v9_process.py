from ikomia import utils, core, dataprocess
import copy
from ikomia.core.task import TaskParam
from ikomia.dnn import dnntrain
import sys
import argparse
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import shutil
import yaml
import numpy as np
import torch
import torch.distributed as dist
from train_yolo_v9.ikutils import prepare_dataset, download_model

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from train_yolo_v9.yolov9.utils.callbacks import Callbacks
from train_yolo_v9.yolov9.utils.downloads import  is_url
from train_yolo_v9.yolov9.utils.general import (LOGGER, check_file, check_yaml, colorstr,
                           get_latest_run, increment_path, print_args, print_mutation)

from train_yolo_v9.yolov9.utils.loggers.comet.comet_utils import check_comet_resume
from train_yolo_v9.yolov9.utils.metrics import fitness
from train_yolo_v9.yolov9.utils.plots import plot_evolve
from train_yolo_v9.yolov9.utils.torch_utils import  select_device
from train_yolo_v9.yolov9.train_dual import train

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = None#check_git_info()


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class TrainYoloV9Param(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        # Create models folder
        models_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
        dataset_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset")
        os.makedirs(models_folder, exist_ok=True)
        os.makedirs(dataset_folder, exist_ok=True)
        self.cfg["dataset_folder"] = dataset_folder
        self.cfg["model_name"] = "yolov9-c"
        self.cfg["model_weight_file"] = ""
        self.cfg["epochs"] = 50
        self.cfg["batch_size"] = 8
        self.cfg["train_imgsz"] = 640
        self.cfg["test_imgsz"] = 640
        self.cfg["dataset_split_ratio"] = 0.9
        self.cfg["config_file"] = ""
        self.cfg["output_folder"] = os.path.dirname(os.path.realpath(__file__)) + "/runs/"

    def set_values(self, param_map):
        self.cfg["dataset_folder"] = param_map["dataset_folder"]
        self.cfg["model_name"] = param_map["model_name"]
        self.cfg["model_weight_file"] = param_map["model_weight_file"]
        self.cfg["epochs"] = int(param_map["epochs"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["train_imgsz"] = int(param_map["train_imgsz"])
        self.cfg["test_imgsz"] = int(param_map["test_imgsz"])
        self.cfg["dataset_split_ratio"] = float(param_map["dataset_split_ratio"])
        self.cfg["config_file"] = param_map["config_file"]
        self.cfg["output_folder"] = param_map["output_folder"]

# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class TrainYoloV9(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)


        # Create parameters object
        if param is None:
            self.set_param_object(TrainYoloV9Param())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.enable_mlflow(True)
        self.enable_tensorboard(True)
        self.device = None
        self.opt = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def run(self):
        param = self.get_param_object()
        dataset_input = self.get_input(0)

        # Conversion from Ikomia dataset to YoloV5
        print("Preparing dataset...")
        dataset_yaml = prepare_dataset(dataset_input, param.cfg["dataset_folder"],
                                       param.cfg["dataset_split_ratio"])
  
        print("Collecting configuration parameters...")
        self.opt = self.load_config(dataset_yaml)

        # Call begin_task_run for initialization
        self.begin_task_run()

        print("Start training...")
        self.start_training()

        # Copy Past dataset class info for inference
        source_file_path = dataset_yaml
        destination_file_path = os.path.join(self.opt.save_dir, 'classes.yaml')
        shutil.copyfile(source_file_path, destination_file_path)

        # Call end_task_run to finalize process
        self.end_task_run()

    def load_config(self, dataset_yaml):
        param = self.get_param_object()

        if len(sys.argv) == 0:
                sys.argv = ["ikomia"]
        parser = argparse.ArgumentParser()
        # parser.add_argument('--weights', type=str, default=ROOT / 'yolo.pt', help='initial weights path')
        # parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
        parser.add_argument('--weights', type=str, default='', help='initial weights path')
        parser.add_argument('--cfg', type=str, default='yolo.yaml', help='model.yaml path')
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='dataset.yaml path')
        parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-high.yaml', help='hyperparameters path')
        parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
        parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
        parser.add_argument('--rect', action='store_true', help='rectangular training')
        parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        parser.add_argument('--noval', action='store_true', help='only validate final epoch')
        parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
        parser.add_argument('--noplots', action='store_true', help='save no plot files')
        parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
        parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
        parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
        parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'LION'], default='SGD', help='optimizer')
        parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
        parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
        parser.add_argument('--name', default='exp', help='save to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--quad', action='store_true', help='quad dataloader')
        parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
        parser.add_argument('--flat-cos-lr', action='store_true', help='flat cosine LR scheduler')
        parser.add_argument('--fixed-lr', action='store_true', help='fixed LR scheduler')
        parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
        parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
        parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
        parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
        parser.add_argument('--seed', type=int, default=0, help='Global training seed')
        parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
        parser.add_argument('--min-items', type=int, default=0, help='Experimental')
        parser.add_argument('--close-mosaic', type=int, default=0, help='Experimental')

        # Logger arguments
        parser.add_argument('--entity', default=None, help='Entity')
        parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
        parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
        parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "yolov9", "models", "detect",
                                param.cfg["model_name"] + ".yaml")


        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            parser.set_defaults(**config)

        opt = parser.parse_args(args=[])
        opt.data = dataset_yaml
        opt.cfg = config_path
        # Override with GUI parameters
        if param.cfg["config_file"]:
            opt.hyp = param.cfg["config_file"]
        else:
            opt.hyp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov9", "data", "hyps", "hyp.scratch-high.yaml")

        models_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
        opt.weights = param.cfg["model_weight_file"] if param.cfg["model_weight_file"] != "" else \
            os.path.join(models_folder, param.cfg["model_name"] + ".pt")
        if not os.path.isfile(opt.weights):
            download_model(param.cfg["model_name"], models_folder)
        opt.epochs = param.cfg["epochs"]
        opt.batch_size = param.cfg["batch_size"]
        opt.img_size = [param.cfg["train_imgsz"], param.cfg["test_imgsz"]]
        opt.project = param.cfg["output_folder"]
        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
        opt.name = str_datetime
        opt.tb_dir = str((Path(core.config.main_cfg["tensorboard"]["log_uri"]) / opt.name))
        opt.stop_train = False

        if sys.platform == 'win32':
            opt.workers = 0

        return opt

    def start_training(self, callbacks=Callbacks()):
                # Checks
        if RANK in {-1, 0}:
            print_args(vars(self.opt))
            #check_git_status()
            #check_requirements()

        # Resume (from specified or most recent last.pt)
        if self.opt.resume and not check_comet_resume(self.opt) and not self.opt.evolve:
            last = Path(check_file(self.opt.resume) if isinstance(self.opt.resume, str) else get_latest_run())
            opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
            opt_data = self.opt.data  # original dataset
            if opt_yaml.is_file():
                with open(opt_yaml, errors='ignore') as f:
                    d = yaml.safe_load(f)
            else:
                d = torch.load(last, map_location='cpu')['opt']
            self.opt = argparse.Namespace(**d)  # replace
            self.opt.cfg, self.opt.weights, self.opt.resume = '', str(last), True  # reinstate
            if is_url(opt_data):
                self.opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
        else:
            self.opt.data, self.opt.cfg, self.opt.hyp, self.opt.weights, self.opt.project = \
                check_file(self.opt.data), check_yaml(self.opt.cfg), check_yaml(self.opt.hyp), str(self.opt.weights), str(self.opt.project)  # checks
            assert len(self.opt.cfg) or len(self.opt.weights), 'either --cfg or --weights must be specified'
            if self.opt.evolve:
                if self.opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                    self.opt.project = str(ROOT / 'runs/evolve')
                self.opt.exist_ok, self.opt.resume = self.opt.resume, False  # pass resume to exist_ok and disable resume
            if self.opt.name == 'cfg':
                self.opt.name = Path(self.opt.cfg).stem  # use model.yaml as name
            self.opt.save_dir = str(increment_path(Path(self.opt.project) / self.opt.name, exist_ok=self.opt.exist_ok))    
            
        # DDP mode
        device = select_device(self.opt.device, batch_size=self.opt.batch_size)
        if LOCAL_RANK != -1:
            msg = 'is not compatible with YOLO Multi-GPU DDP training'
            assert not self.opt.image_weights, f'--image-weights {msg}'
            assert not self.opt.evolve, f'--evolve {msg}'
            assert self.opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
            assert self.opt.batch_size % WORLD_SIZE == 0, f'--batch-size {self.opt.batch_size} must be multiple of WORLD_SIZE'
            assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
            torch.cuda.set_device(LOCAL_RANK)
            device = torch.device('cuda', LOCAL_RANK)
            dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

        # Train
        if not self.opt.evolve:
            train(self.opt.hyp, self.opt, device, callbacks)

        # Evolve hyperparameters (optional)
        else:
            # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
            meta = {
                'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

            with open(self.opt.hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
                if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                    hyp['anchors'] = 3
            if self.opt.noautoanchor:
                del hyp['anchors'], meta['anchors']
            self.opt.noval, self.opt.nosave, save_dir = True, True, Path(self.opt.save_dir)  # only val/save final epoch
            # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
            evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
            if self.opt.bucket:
                os.system(f'gsutil cp gs://{self.opt.bucket}/evolve.csv {evolve_csv}')  # download evolve.csv if exists

            for _ in range(self.opt.evolve):  # generations to evolve
                if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                    # Select parent(s)
                    parent = 'single'  # parent selection method: 'single' or 'weighted'
                    x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                    n = min(5, len(x))  # number of previous results to consider
                    x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                    w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                    if parent == 'single' or len(x) == 1:
                        # x = x[random.randint(0, n - 1)]  # random selection
                        x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                    elif parent == 'weighted':
                        x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                    # Mutate
                    mp, s = 0.8, 0.2  # mutation probability, sigma
                    npr = np.random
                    npr.seed(int(time.time()))
                    g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                    ng = len(meta)
                    v = np.ones(ng)
                    while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                        v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                    for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                        hyp[k] = float(x[i + 7] * v[i])  # mutate

                # Constrain to limits
                for k, v in meta.items():
                    hyp[k] = max(hyp[k], v[1])  # lower limit
                    hyp[k] = min(hyp[k], v[2])  # upper limit
                    hyp[k] = round(hyp[k], 5)  # significant digits

                # Train mutation
                results = train(hyp.copy(), self.opt, device, callbacks)
                callbacks = Callbacks()
                # Write mutation results
                keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss',
                        'val/obj_loss', 'val/cls_loss')
                print_mutation(keys, results, hyp.copy(), save_dir, self.opt.bucket)

            # Plot results
            plot_evolve(evolve_csv)
            LOGGER.info(f'Hyperparameter evolution finished {self.opt.evolve} generations\n'
                        f"Results saved to {colorstr('bold', save_dir)}\n"
                        f'Usage example: $ python train.py --hyp {evolve_yaml}')


    def stop(self):
        super().stop()
        self.opt.stop_train = True

# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class TrainYoloV9Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "train_yolo_v9"
        self.info.short_description = "Train YOLOv9 models"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Wang, Chien-Yao  and Liao, Hong-Yuan Mark"
        self.info.article = "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information"
        self.info.journal = "arXiv:2402.13616"
        self.info.year = 2024
        self.info.license = "GNU General Public License v3.0"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/abs/2402.13616"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/train_yolo_v9"
        self.info.original_repository = "https://github.com/WongKinYiu/yolov9"
        # Keywords used for search
        self.info.keywords = "YOLO, object, detection, real-time, Pytorch"
        self.info.algo_type = core.AlgoType.TRAIN
        self.info.algo_tasks = "OBJECT_DETECTION"

    def create(self, param=None):
        # Create algorithm object
        return TrainYoloV9(self.info.name, param)


if __name__ == '__main__':
    param = TrainYoloV9Param()
    train_yolo_v9_process = TrainYoloV9("train_yolo_v9", param)
    train_yolo_v9_process.run()