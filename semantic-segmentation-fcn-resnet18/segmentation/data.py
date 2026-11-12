"""PASCAL VOC 2012 semantic segmentation data loader (official train/val lists)."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

class VOCSegmentationDataset(Dataset):
    def __init__(self, voc_root, split="train", image_size=128):
        self.root = Path(voc_root).expanduser()
        if split not in ("train", "val"):
            raise ValueError("split must be train or val")
        split_path = self.root / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"VOC split list not found: {split_path}")
        self.ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        if not self.ids:
            raise ValueError(f"No image IDs found in {split_path}")
        self.image_size = (image_size, image_size)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id_ = self.ids[index]
        img_path = self.root / "JPEGImages" / f"{id_}.jpg"
        mask_path = self.root / "SegmentationClass" / f"{id_}.png"
        with Image.open(img_path) as im:
            image = TF.resize(im.convert("RGB"), self.image_size, interpolation=InterpolationMode.BILINEAR)
            image = TF.to_tensor(image)
        with Image.open(mask_path) as im:
            # VOC PNG is palette-indexed: retain class indices 0..20 and ignore ID 255.
            if im.mode != "P":
                raise ValueError(f"Expected VOC indexed PNG mask (mode P): {mask_path}")
            mask = TF.resize(im, self.image_size, interpolation=InterpolationMode.NEAREST)
            mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask

def build_loader(root, split, image_size=128, batch_size=4, workers=0):
    dataset = VOCSegmentationDataset(root, split, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                      num_workers=workers, pin_memory=torch.cuda.is_available())
