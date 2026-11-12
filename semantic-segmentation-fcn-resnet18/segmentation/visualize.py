"""Save side-by-side original / target / prediction for a trained model."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from .model import FCNResNet18
from .data import VOCSegmentationDataset
from .utils import choose_device

# Standard PASCAL VOC class color palette (background + 20 classes).
VOC_COLORS=np.array([[0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],
 [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
 [64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128],
 [192,128,128],[0,64,0],[128,64,0],[0,192,0],[128,192,0],[0,64,128]],dtype=np.uint8)

def color_mask(values):
    array=np.asarray(values)
    rgb=np.zeros((*array.shape,3),dtype=np.uint8)
    valid=(array>=0)&(array<len(VOC_COLORS))
    rgb[valid]=VOC_COLORS[array[valid]]
    return Image.fromarray(rgb)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--voc-root",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--count",type=int,default=6)
    p.add_argument("--out-dir",default="outputs/visualizations")
    p.add_argument("--device",choices=["auto","cpu","cuda","mps"],default="auto")
    args=p.parse_args()
    ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    model=FCNResNet18(num_classes=ckpt.get("num_classes",21),pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    device=choose_device(args.device)
    model.to(device).eval()
    dataset=VOCSegmentationDataset(args.voc_root,"val",ckpt["image_size"])
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for i in range(min(args.count,len(dataset))):
            image,mask=dataset[i]
            pred=model(image.unsqueeze(0).to(device)).argmax(1).squeeze(0).cpu().numpy()
            input_img=Image.fromarray((image.permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8))
            panels=[input_img,color_mask(mask.numpy()),color_mask(pred)]
            w,h=input_img.size
            result=Image.new("RGB",(w*3,h+26),"white")
            draw=ImageDraw.Draw(result)
            for j,(panel,title) in enumerate(zip(panels,["Input","Ground truth","Prediction"])):
                result.paste(panel,(j*w,26));draw.text((j*w+6,6),title,fill="black")
            result.save(out/f"sample_{i:02d}.png")
    print(f"Saved visualizations to {out}")
if __name__=="__main__":main()
