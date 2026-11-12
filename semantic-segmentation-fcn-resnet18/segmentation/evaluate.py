"""Evaluate a trained checkpoint on VOC validation split."""
import argparse
import json
import torch
from .model import FCNResNet18
from .data import build_loader
from .utils import choose_device
from .metrics import SegmentationMetrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--voc-root",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--workers",type=int,default=0)
    p.add_argument("--device",choices=["auto","cpu","cuda","mps"],default="auto")
    args=p.parse_args()
    device=choose_device(args.device)
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    model=FCNResNet18(pretrained=False,num_classes=checkpoint.get("num_classes",21))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    loader=build_loader(args.voc_root,"val",checkpoint["image_size"],args.batch_size,args.workers)
    metrics=SegmentationMetrics(num_classes=checkpoint.get("num_classes",21))
    with torch.inference_mode():
        for images,masks in loader:
            logits=model(images.to(device))
            metrics.update(logits.argmax(1),masks)
    print(json.dumps(metrics.compute(),indent=2))
if __name__=="__main__":main()
