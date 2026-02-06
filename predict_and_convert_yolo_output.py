import json
from ultralytics import YOLO
import yaml
import dvc.api

import preprocess_data


def predict(ds, model_path_input, conf=0.1):
    predictions_folder = ds.path_to_dataset.joinpath('predictions')
    if not predictions_folder.exists():
        model_path = model_path_input
        model = YOLO(model_path)
        if ds.images_folder.exists():
            results_list = model(source=str(ds.images_folder),
                                 project=str(ds.path_for_data),
                                 name=predictions_folder.name,
                                 stream=True,
                                 save=False, show=False, save_conf=True, save_txt=True, conf=conf,
                                 save_crop=False, agnostic_nms=True)

            for r in results_list:
                pass
    else:
        print('Using existing predictions folder. Delete if you wanted to overwrite')

    return predictions_folder


if __name__ == '__main__':
    path_to_dataset = '/albedo/work/projects/p_OZA_AI/YOLO/Data/validation'
    model_path_input = '/albedo/work/projects/p_OZA_AI/YOLO/runs/biodcase_baseline/train9/weights/best.pt'

    config_path = './dataset_config.json'
    f = open(config_path)
    config = json.load(f)

    with open('./custom_joined.yaml', 'r') as file:
        yolo_config = yaml.safe_load(file)

    yolo_ds = preprocess_data.YOLODataset(config, path_to_dataset,path_to_dataset)
    output_folder = predict(yolo_ds,model_path_input)
    yolo_ds.convert_yolo_detections_to_csv(output_folder, reverse_class_encoding=yolo_config['names'])
