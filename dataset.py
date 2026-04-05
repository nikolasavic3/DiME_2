# dataset.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CELEBA_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young"
]


class CelebADataset(Dataset):

    def __init__(self, root, attr="Smiling", split="train", size=256):
        self.root = root
        self.img_dir = os.path.join(root, "img_align_celeba")

        # Load attributes — CSV with image_id as index
        attr_path = os.path.join(root, "list_attr_celeba.csv")
        attrs = pd.read_csv(attr_path, index_col="image_id")
        attrs = (attrs + 1) // 2  # -1/1 → 0/1

        # Load split — CSV with image_id and partition columns
        split_path = os.path.join(root, "list_eval_partition.csv")
        split_df = pd.read_csv(split_path, index_col="image_id")

        # 0=train, 1=val, 2=test
        split_map = {"train": 0, "val": 1, "test": 2}
        mask = split_df["partition"] == split_map[split]
        self.filenames = split_df[mask].index.values
        self.labels = attrs.loc[self.filenames, attr].values

        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])

        pos = self.labels.sum()
        neg = len(self.labels) - pos
        print(f"CelebA {split}: {len(self.filenames)} images  |  "
              f"{attr}: {pos} positive  {neg} negative")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = int(self.labels[idx])
        return img, label