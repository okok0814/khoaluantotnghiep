from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import PIL.Image
import torch
from torch.utils.data import Dataset


class FashionIQTripletDataset(Dataset):
    """
    "Loader này được thiết kế để điều chỉnh file FashionIQ CSV của nhóm
    (yêu cầu đủ 4 cột: candidate, modifier, target, category)
    sao cho tương thích trơn tru với quy trình xử lý (workflow) của UniFashion.
    Điểm mạnh của loader này là tính linh hoạt không bị phụ thuộc
    vào định dạng chú thích riêng rườm rà của UniFashion
    và loại bỏ hoàn toàn việc hardcode đường dẫn máy cá nhân.
    Khi hoạt động, hệ thống sẽ tự động gọi các file ảnh `.jpg` 
    đã tải từ FashionIQ và trích xuất ra đúng 3 thành phần cốt lõi
    để đưa vào mô hình: ảnh gốc (candidate), ảnh đích (target) và
    câu văn mô tả (modification text)."
    """

    REQUIRED_COLUMNS = {"candidate", "modifier", "target", "category"}

    def __init__(
        self,
        csv_path,
        images_root,
        preprocess=None,
        category: Optional[str] = None,
        strict: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.images_root = Path(images_root)
        self.preprocess = preprocess
        self.category = category
        self.strict = strict

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Không tìm thấy CSV: {self.csv_path}")

        if not self.images_root.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {self.images_root}")

        self.data = pd.read_csv(
            self.csv_path,
            dtype={
                "candidate": str,
                "modifier": str,
                "target": str,
                "category": str,
            },
        )

        missing_columns = self.REQUIRED_COLUMNS - set(self.data.columns)
        if missing_columns:
            raise ValueError(
                f"CSV thiếu các cột bắt buộc {sorted(missing_columns)}"
            )

        if category is not None:
            valid_categories = {"dress", "shirt", "toptee"}
            if category not in valid_categories:
                raise ValueError(
                    f"category phải thuộc {sorted(valid_categories)}"
                )

            self.data = (
                self.data[self.data["category"].eq(category)]
                .reset_index(drop=True)
            )

        # Train/val phải có target. Test FashionIQ có thể thiếu target.
        if self.strict and self.data["target"].isna().any():
            raise ValueError(
                "Dataset đang strict=True nhưng có target bị thiếu. "
                "Dùng train/val hoặc đặt strict=False cho test."
            )

    def __len__(self):
        return len(self.data)

    def _image_path(self, image_id: str) -> Path:
        """
        FashionIQ của nhóm dùng ảnh .jpg.
        Có fallback .jpeg/.png để loader dễ kiểm tra dữ liệu.
        """
        image_id = str(image_id)

        candidates = [
            self.images_root / f"{image_id}.jpg",
            self.images_root / f"{image_id}.jpeg",
            self.images_root / f"{image_id}.png",
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Không tìm thấy ảnh cho image_id={image_id}. "
            f"Đã kiểm tra {[str(p) for p in candidates]}"
        )

    @staticmethod
    def _open_rgb(path: Path):
        return PIL.Image.open(path).convert("RGB")

    def __getitem__(self, index) -> Dict[str, Any]:
        row = self.data.iloc[index]

        candidate_id = str(row["candidate"])
        modifier = str(row["modifier"])
        category = str(row["category"])

        candidate_path = self._image_path(candidate_id)
        candidate_image = self._open_rgb(candidate_path)

        target_value = row["target"]
        target_id = None
        target_path = None
        target_image = None

        if pd.notna(target_value) and str(target_value).strip():
            target_id = str(target_value)
            target_path = self._image_path(target_id)
            target_image = self._open_rgb(target_path)

        if self.preprocess is not None:
            candidate_image = self.preprocess(candidate_image)

            if target_image is not None:
                target_image = self.preprocess(target_image)

        return {
            "reference_image": candidate_image,
            "target_image": target_image,
            "modifier": modifier,
            "candidate_id": candidate_id,
            "target_id": target_id,
            "category": category,
            "candidate_path": str(candidate_path),
            "target_path": str(target_path) if target_path else None,
        }


def unifashion_collate_fn(batch):
    """
    Hàm này có nhiệm vụ tự động gộp các hình ảnh đã qua tiền xử lý
    (đang ở dạng tensor) thành từng lô (batch) để đưa vào mô hình tính toán,
    đồng thời giữ nguyên phần văn bản và metadata ở định dạng list
    nhằm thuận tiện cho việc truy xuất và rà soát thông tin đi kèm.
    """
    reference_images = [item["reference_image"] for item in batch]
    target_images = [item["target_image"] for item in batch]

    if all(torch.is_tensor(x) for x in reference_images):
        reference_images = torch.stack(reference_images)

    if all(torch.is_tensor(x) for x in target_images if x is not None) and all(
        x is not None for x in target_images
    ):
        target_images = torch.stack(target_images)

    return {
        "reference_images": reference_images,
        "target_images": target_images,
        "modifiers": [item["modifier"] for item in batch],
        "candidate_ids": [item["candidate_id"] for item in batch],
        "target_ids": [item["target_id"] for item in batch],
        "categories": [item["category"] for item in batch],
        "candidate_paths": [item["candidate_path"] for item in batch],
        "target_paths": [item["target_path"] for item in batch],
    }