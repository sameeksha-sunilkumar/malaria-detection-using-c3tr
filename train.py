import torch
torch.cuda.set_per_process_memory_fraction(0.7) 
torch.cuda.empty_cache()

import sys
import os
from pathlib import Path

project_root = Path("/content/drive/MyDrive/malaria detection")
sys.path.insert(0, str(project_root))

try:
    from models.custom_modules.c3tr import C3TR
    from ultralytics import YOLO
    from ultralytics.nn.modules import C3 as ultralytics_C3
except ImportError as e:
    print(f"Import error: {e}")
    print("\nFiles in models/custom_modules:")
    custom_modules_path = project_root / 'models' / 'custom_modules'
    print(os.listdir(custom_modules_path) if custom_modules_path.exists() else "Directory not found")
    raise

ultralytics_C3.C3TR = C3TR

if __name__ == "__main__":

    from roboflow import Roboflow
    rf = Roboflow(api_key="JIpWla822HdxFfIQbun5")
    project = rf.workspace("malaria-qovy0").project("new-malaria-acgtb")
    dataset = project.version(4).download("yolov5", location="/content/datasets")
                

    #model_config = project_root / 'models' / 'custom_yolov5s.yaml' 
    #model_config = 'yolov5m.yaml'
    model_config = 'yolov5l.yaml'
    dataset_config = "/content/datasets/data.yaml"

  
    print(f"Using model config: {model_config}")
    print(f"Using dataset config: {dataset_config}")
    model = YOLO(str(model_config))
    
    results = model.train(
        data=dataset_config,
        epochs=200,
        batch=4,  
        imgsz=416,
        device='0',
        name='new_malaria_v1_t4',  
        amp=False,
        mosaic=0.1,  
        workers=2,
        exist_ok=True,
        verbose=True,  
        patience=50,   
     
        lr0=0.0005,      
        lrf=0.01,      
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
  
        box=0.05,
        cls=0.5,
        dfl=1.5,

        hsv_h=0.015,   
        hsv_s=0.7,     
        hsv_v=0.4,     
        degrees=5.0,  
        translate=0.05, 
        scale=0.3,     
        shear=1.0,     
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,   
        mixup=0.2,    
        copy_paste=0.2
    )
