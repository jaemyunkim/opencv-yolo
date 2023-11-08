import os
from pathlib import Path
import requests


def get_yolo_weights(model_name):
    model_name = model_name.lower()

    # select YOLO version
    if model_name.startswith("yolov1"):
        pass
    elif model_name.startswith("yolov2"):
        pass
    elif model_name.startswith("yolov3"):
        urls = [
            f"https://pjreddie.com/media/files/{model_name}.weights", # weights
            f"https://github.com/pjreddie/darknet/raw/master/cfg/{model_name}.cfg", # cfg
        ]
    else:
        print(f"not found the currect model name: {model_name}")

    # download model files
    for url in urls:
        target_file = Path(f"models/{Path(url).name}")
        if not target_file.exists():
            target_file.parent.mkdir(parents=True, exist_ok=True)   # create directory
            r = requests.get(url) # create HTTP response object
            with open(str(target_file), "wb") as f: 
                # Saving received content as a png file in 
                # binary format 
            
                # write the contents of the response (r.content) 
                # to a new file in binary mode. 
                f.write(r.content)
            
            print(f"download file: {str(target_file)}")
        else:
            print(f"found file: {str(target_file)}")
