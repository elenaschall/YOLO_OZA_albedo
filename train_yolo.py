try:
    import comet_ml
except ModuleNotFoundError:
    comet_ml = None
import yaml
import torch
torch.cuda.empty_cache()
from ultralytics import YOLO
import os
#import dvc.api
import mlflow


def run():
    YAML_FILE = './custom_joined.yaml'
    run_name = 'biodcase_baseline' # Change to the name of your run

    # Check if CUDA is available
    print('CUDA device count:')
    print(torch.cuda.device_count())

    # Read the config file
    with open(YAML_FILE, 'r') as file:
        config = yaml.safe_load(file)

    if "COMET_API_KEY" in os.environ and comet_ml is not None:
        experiment = comet_ml.Experiment(
            project_name="biodcase",
        )

    # Load a model
    model = YOLO('yolo11s.pt')

    # train the model
    best_params = {
        'iou': 0.3,
        'imgsz': 640,
        'rect': False,
        'hsv_s': 0,
        'hsv_v':  0,
        'degrees': 0,
        'translate': 0,
        'scale': 0,
        'shear': 0,
        'perspective': 0,
        'flipud': 0,
        'fliplr': 0,
        'bgr': 0,
        'mosaic': 0,
        'mixup': 0,
        'copy_paste': 0,
        'erasing': 0,
        'crop_fraction': 0,
    }
    torch.cuda.memory_summary(device=None, abbreviated=False)

    model.train(epochs=20, batch=32, data=YAML_FILE, device=[0,1],
                project=config['path'] + '/runs/' + run_name, resume=False,plots=True, **best_params)

    if "COMET_API_KEY" in os.environ and comet_ml is not None:
        experiment.end()


if __name__ == '__main__':
    run()
