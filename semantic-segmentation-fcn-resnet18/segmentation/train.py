"""Train FCN-ResNet18 with VOC official training split."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from .data import build_loader
from .model import FCNResNet18
from .metrics import SegmentationMetrics
from .utils import choose_device, seed_everything

def run(args):
    seed_everything(args.seed)
    device = choose_device(args.device)
    train_loader = build_loader(args.voc_root, "train", args.image_size, args.batch_size, args.workers)
    val_loader = build_loader(args.voc_root, "val", args.image_size, args.batch_size, args.workers)
    model = FCNResNet18(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss, steps = 0., 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); steps += 1
        model.eval()
        metrics = SegmentationMetrics()
        with torch.inference_mode():
            for images, masks in val_loader:
                logits = model(images.to(device))
                metrics.update(logits.argmax(dim=1), masks)
        record = {"epoch":epoch, "training_loss":total_loss/max(steps,1), **metrics.compute()}
        print(json.dumps(record))
        history.append(record)
        torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,"image_size":args.image_size,
                    "num_classes":21},output_dir/"last.pt")
    (output_dir/"history.json").write_text(json.dumps(history,indent=2)+"\n")
    print(f"Saved checkpoint and history to {output_dir}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--voc-root",required=True,help="Path to VOCdevkit/VOC2012")
    p.add_argument("--epochs",type=int,default=1)
    p.add_argument("--image-size",type=int,default=128)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--workers",type=int,default=0)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",choices=["auto","cpu","cuda","mps"],default="auto")
    p.add_argument("--no-pretrained",action="store_true",help="Avoid downloading pretrained ResNet-18 weights")
    p.add_argument("--output-dir",default="outputs")
    run(p.parse_args())
if __name__=="__main__":main()
