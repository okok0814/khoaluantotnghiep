import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


TRIPLETS_CSVS = [
    "./data/processed/fashioniq_triplets_train.csv",  # 18.000 dòng, đủ target
    "./data/processed/fashioniq_triplets_val.csv",    # 6.016 dòng, đủ target
    "./data/processed/fashioniq_triplets_test.csv",   # 6.118 dòng, target RỖNG (chuẩn)
]
IMAGES_DIR = "./data/fashioniq/resized_images/resized_images"

OUTPUT_DIR = "./logs/thesis_embeddings"
BATCH_SIZE = 64
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CATEGORY_SUBDIRS = ("dress", "shirt", "toptee")  # thử các thư mục con này nếu tìm phẳng thất bại


def gather_unique_image_ids(triplet_csvs):

    ids = set()
    for path in triplet_csvs:
        if not os.path.exists(path):
            print(f"[CẢNH BÁO] Không thấy {path}, bỏ qua.")
            continue
        df = pd.read_csv(path)
        for col in ("candidate", "target"):
            if col not in df.columns:
                print(f"[CẢNH BÁO] {path} không có cột '{col}' — kiểm tra lại tên cột thật.")
                continue
            n_null = int(df[col].isna().sum())
            if n_null:
                print(f"[{os.path.basename(path)}] cột '{col}': {n_null}/{len(df)} dòng trống "
                      f"(bỏ qua khi gom ID, không tính vào embedding)")
            ids.update(df[col].dropna().astype(str).tolist())
    return sorted(ids)


def resolve_image_path(image_id, images_dir=IMAGES_DIR):
    """Tìm file ảnh thật cho một image_id: thử phẳng trước, rồi thử trong các
    thư mục con category. Trả None nếu không tìm thấy (ảnh thiếu)."""
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(images_dir, f"{image_id}{ext}")
        if os.path.exists(p):
            return p
        for cat in CATEGORY_SUBDIRS:
            p = os.path.join(images_dir, cat, f"{image_id}{ext}")
            if os.path.exists(p):
                return p
    return None


# ---------------------------------------------------------------------------
# 2. Dataset đọc ảnh theo batch — không để 1 ảnh thiếu làm crash cả batch
# ---------------------------------------------------------------------------
class ImageIdDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, images_dir, transform):
        self.image_ids = image_ids
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = resolve_image_path(image_id, self.images_dir)
        if path is None:
            return torch.zeros(3, 224, 224), image_id, False
        img = Image.open(path).convert("RGB")
        return self.transform(img), image_id, True


# ---------------------------------------------------------------------------
# 3. Hai backbone
# ---------------------------------------------------------------------------
def build_resnet50():
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()  # bỏ lớp phân loại 1000-class -> lấy thẳng vector 2048-d sau pooling
    model.eval().to(DEVICE)
    transform = weights.transforms()  # dùng đúng tiền xử lý mà checkpoint này được huấn luyện
    return model, transform, 2048


def build_clip():
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def transform(pil_img):
        return processor(images=pil_img, return_tensors="pt")["pixel_values"][0]

    return model, transform, 512


# ---------------------------------------------------------------------------
# 4. Trích xuất theo batch + log tiến độ
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract(model, backbone_name, image_ids, transform, dim, batch_size=BATCH_SIZE):
    dataset = ImageIdDataset(image_ids, IMAGES_DIR, transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=NUM_WORKERS
    )

    all_vecs = np.zeros((len(image_ids), dim), dtype=np.float32)
    all_ok = np.zeros(len(image_ids), dtype=bool)
    seen = 0
    t0 = time.time()

    for batch_imgs, _batch_ids, batch_valid in loader:
        batch_imgs = batch_imgs.to(DEVICE)
        if backbone_name == "clip":
            out = model.get_image_features(pixel_values=batch_imgs)
            feats = out.pooler_output if hasattr(out, "pooler_output") else out
        else:
            feats = model(batch_imgs)
        feats = feats.float().cpu().numpy()
        assert feats.shape[-1] == dim, (
            f"[{backbone_name}] Vector trả về có {feats.shape[-1]} chiều, khác {dim} chiều "
            f"kỳ vọng — có thể do phiên bản thư viện thay đổi API. Kiểm tra lại trước khi dùng "
            f"kết quả này."
        )

        n = len(feats)
        all_vecs[seen:seen + n] = feats
        all_ok[seen:seen + n] = batch_valid.numpy()
        seen += n

    n_missing = int((~all_ok).sum())
    print(f"[{backbone_name}] Đã trích xuất {seen}/{len(image_ids)} ảnh "
          f"({n_missing} ảnh thiếu file) trong {time.time() - t0:.1f}s")
    return all_vecs, all_ok


def save_embeddings(backbone_name, image_ids, vecs, ok_mask, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{backbone_name}_image_embeddings.npz")
    np.savez_compressed(out_path, ids=np.array(image_ids), embeddings=vecs, valid=ok_mask)
    print(f"[{backbone_name}] Đã lưu: {out_path}  shape={vecs.shape}")
    return out_path


def run(triplet_csvs=TRIPLETS_CSVS, images_dir=IMAGES_DIR, output_dir=OUTPUT_DIR):
    image_ids = gather_unique_image_ids(triplet_csvs)
    print(f"Tổng số image_id duy nhất cần trích xuất: {len(image_ids)}")
    assert len(image_ids) > 0, (
        "Không tìm thấy image_id nào — kiểm tra lại đường dẫn CSV / tên cột 'candidate'/'target'."
    )

    for backbone_name, builder in [("resnet50", build_resnet50), ("clip", build_clip)]:
        model, transform, dim = builder()
        vecs, ok = extract(model, backbone_name, image_ids, transform, dim)
        save_embeddings(backbone_name, image_ids, vecs, ok, output_dir)



def _self_test():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        images_dir = os.path.join(tmp, "images")
        os.makedirs(images_dir)
        fake_ids = [f"img{i}" for i in range(4)]
        for fid in fake_ids:
            Image.new("RGB", (32, 32), color=(hash(fid) % 255, 0, 0)).save(
                os.path.join(images_dir, f"{fid}.jpg")
            )

        # img_missing chỉ xuất hiện trong CSV, không có file ảnh -> kiểm tra xử lý ảnh thiếu
        df_train = pd.DataFrame({
            "candidate": ["img0", "img1", "img2"],
            "target": ["img1", "img2", "img_missing"],
        })
        train_csv = os.path.join(tmp, "triplets_train.csv")
        df_train.to_csv(train_csv, index=False)

        # mô phỏng ĐÚNG tình huống thật của test.csv: cột target rỗng 100%
        df_test = pd.DataFrame({
            "candidate": ["img3"],
            "target": [None],
        })
        test_csv = os.path.join(tmp, "triplets_test.csv")
        df_test.to_csv(test_csv, index=False)

        ids = gather_unique_image_ids([train_csv, test_csv])
        # 'target' rỗng của test.csv KHÔNG được biến thành id rác kiểu "nan"
        assert set(ids) == {"img0", "img1", "img2", "img_missing", "img3"}, ids
        assert "nan" not in ids and float("nan") not in ids

        # model giả: không tải trọng số thật, chỉ lấy trung bình pixel -> vector 3 chiều
        class FakeModel(nn.Module):
            def forward(self, x):
                return x.mean(dim=[2, 3])

        fake_transform = transforms.Compose([transforms.ToTensor()])
        global IMAGES_DIR
        old_images_dir, IMAGES_DIR = IMAGES_DIR, images_dir
        try:
            vecs, ok = extract(FakeModel().eval(), "fake", ids, fake_transform, dim=3, batch_size=2)
        finally:
            IMAGES_DIR = old_images_dir

        # ids đã sắp xếp = ['img0','img1','img2','img3','img_missing'] (5 phần tử);
        # 4 ảnh đầu có file thật, riêng img_missing không có file -> phải bị đánh dấu False
        assert vecs.shape == (5, 3), vecs.shape
        assert ok.tolist() == [True, True, True, True, False], ok

        out_dir = os.path.join(tmp, "out")
        path = save_embeddings("fake", ids, vecs, ok, out_dir)
        loaded = np.load(path, allow_pickle=True)
        assert loaded["embeddings"].shape == (5, 3), loaded["embeddings"].shape
        assert list(loaded["ids"]) == ids

    print("[SELF-TEST PASSED] Gom ID, xử lý ảnh thiếu, trích xuất theo batch, lưu/đọc .npz đều đúng.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        run()