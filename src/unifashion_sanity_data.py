"""Caption-aware adapter on top of the group's existing FashionIQ CSV loader."""
import re

from torchvision import transforms
from torchvision.transforms import functional as TF

from src.fashioniq_unifashion_dataset import FashionIQTripletDataset, unifashion_collate_fn


def clean_caption(text):
    # Same normalization as upstream BlipCaptionProcessor (50-word default).
    text = re.sub(r'([.!"()*#:;~])', " ", text.lower())
    text = re.sub(r"\s{2,}", " ", text).rstrip("\n").strip(" ")
    return " ".join(text.split(" ")[:50])


class TargetPad:
    """UniFashion src/data_utils.py TargetPad formula, unchanged."""
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, image):
        width, height = image.size
        if max(width, height) / min(width, height) < self.ratio:
            return image
        scaled = max(width, height) / self.ratio
        hp, vp = max(int((scaled - width) / 2), 0), max(int((scaled - height) / 2), 0)
        return TF.pad(image, [hp, vp, hp, vp], 0, "constant")


def preprocess(config):
    return transforms.Compose([
        TargetPad(config["target_ratio"]),
        transforms.Resize(config["image_size"], interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(config["image_size"]), transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])


class UniFashionSanityDataset(FashionIQTripletDataset):
    def __init__(self, csv_path, images_root, config, require_unique_targets=True):
        super().__init__(csv_path, images_root, preprocess=preprocess(config), strict=True)
        for column in ("reference_caption", "target_caption", "modifier", "candidate", "target"):
            if column not in self.data or self.data[column].isna().any() or self.data[column].str.strip().eq("").any():
                raise ValueError(f"Missing required {column}; run prepare_unifashion_sanity.py")
        if require_unique_targets and self.data["target"].duplicated().any():
            raise ValueError("Sanity subset requires unique targets to avoid false in-batch negatives")

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["reference_caption"] = clean_caption(self.data.iloc[index]["reference_caption"])
        sample["target_caption"] = clean_caption(self.data.iloc[index]["target_caption"])
        sample["modifier"] = clean_caption(sample["modifier"])
        return sample


def collate(batch):
    packed = unifashion_collate_fn(batch)
    return {"image": packed["reference_images"], "target": packed["target_images"],
            "text_input": packed["modifiers"],
            "reference_caption": [sample["reference_caption"] for sample in batch],
            "target_caption": [sample["target_caption"] for sample in batch]}
