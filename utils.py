import os
from pathlib import Path
import requests
from tqdm import tqdm


def get_yolo_weights(model_name):
    model_name = model_name.lower()
    input_size = [416, 416]

    # select YOLO version
    if model_name.startswith("yolov1") or model_name.startswith("yolov2") or model_name.startswith("yolov3"):
        urls = [
            f"https://pjreddie.com/media/files/{model_name}.weights", # weights
            f"https://github.com/pjreddie/darknet/raw/master/cfg/{model_name}.cfg", # cfg
        ]
        input_size = [416, 416]
    elif model_name.startswith("yolov4"):
        urls = [
            f"https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights",
            f"https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.cfg",

        ]
        input_size = [416, 416]
    elif model_name.startswith("yolov5"):
        urls = [
            f"https://github.com/ultralytics/yolov5/releases/download/v7.0/{model_name}.onnx"
        ]
        input_size = [640, 640]
    else:
        print(f"not found the currect model name: {model_name}")
        input_size = [0, 0]

    # download model files
    get_file_from_url(urls, "models")

    return input_size


def get_file_from_url(urls, local_path = ""):
    local_path = Path(local_path)
    filenames = []

    # download model files
    for url in urls:
        # obtain filename by splitting url and getting  
        target_file = local_path / f"{Path(url).name}"
        filenames.append(target_file)

        if not target_file.exists():
            target_file.parent.mkdir(parents=True, exist_ok=True)   # create directory
            
            chunk_size = 1024
            r = requests.get(url, stream=True) # create HTTP response object
            total = int(r.headers.get('content-length', 0))
            with open(str(target_file), "wb") as f, tqdm(
                desc=str(target_file),
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=chunk_size,
            ) as bar: 
                # Saving received content as a png file in 
                # binary format 
            
                # write the contents of the response (r.content) 
                # to a new file in binary mode. 
                for data in r.iter_content(chunk_size=chunk_size):
                    size = f.write(data)
                    bar.update(size)
            
            print(f"download file: {str(target_file)}")
        else:
            print(f"found file: {str(target_file)}")

    return filenames
