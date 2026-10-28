# -*- coding: utf-8 -*-
r"""
sort_with_clip.py
====================
CLIP（ゼロショット分類）で「コスプレイヤーが写っている画像」と
「通行人・会場風景・床・ブース等」を意味ベースで仕分けるスクリプト。

VS Code でこのファイルを開き、下の「設定エリア」を書き換えて ▶(実行) を押すだけで動きます。

■ 出力
  DST_ROOT/
    target/   ... スコア高(>= THRESHOLD_HIGH): コスプレ画像の可能性が高い（元のサブフォルダ構造を維持）
    review/   ... スコア中(THRESHOLD_LOW 〜 THRESHOLD_HIGH): 目視確認用（同上）
    scores.csv ... 全画像のスコアと判定結果（途中経過も随時保存）
  ※ スコア低(< THRESHOLD_LOW)は既定ではコピーしません（ディスク節約）。
    元画像は一切変更しません（コピーのみ）。

■ 事前準備（仮想環境のターミナルで）
    pip install torch torchvision transformers pillow tqdm

  ※ Python 3.14 だと PyTorch が未対応でインストールに失敗する場合があります。
    その場合は Python 3.12 などで別の仮想環境を作ってください。

■ 初回実行時
  CLIPモデル（約600MB）を自動ダウンロードします（要インターネット接続）。
  2回目以降はキャッシュを使います。

■ おすすめの使い方
  1. まず DRY_RUN = True、LIMIT = 300 で試し、scores.csv を見てスコア分布を確認
  2. THRESHOLD_HIGH / THRESHOLD_LOW を調整
  3. DRY_RUN = False、LIMIT = None で本番実行
"""

import csv
import shutil
import sys
from pathlib import Path

# ==================== 設定エリア ====================
SRC_ROOT = r"F:\dataset_from_video"            # 元画像フォルダ（サブフォルダも再帰探索）
DST_ROOT = r"F:\dataset_final"                 # 出力先（target/ review/ scores.csv が作られる）

MODEL_NAME = "openai/clip-vit-base-patch32"   # 精度重視なら "openai/clip-vit-large-patch14"（重い）
BATCH_SIZE = 32                                # GPUメモリが足りない/CPUのみなら 8〜16 に

THRESHOLD_HIGH = 0.60    # これ以上 → target
THRESHOLD_LOW = 0.35     # これ未満 → 除外。間は review
COPY_REJECTED = False    # True にすると除外画像も rejected/ にコピー

DRY_RUN = True           # True: コピーせず scores.csv だけ作る（最初は True 推奨）
LIMIT = 300              # 試験用に処理枚数を制限。本番は None
# ====================================================

# 判定に使う文章（ここを書き換えると判定基準を調整できる）
POSITIVE_PROMPTS = [
    "a photo of a cosplayer wearing a colorful wig and an anime character costume",
    "a photo of a person in an elaborate cosplay costume",
    "a close-up photo of a cosplayer's face with a wig",
]
NEGATIVE_PROMPTS = [
    "a photo of the back of a passerby wearing plain dark clothes",
    "a photo of a crowd of people walking in a convention hall",
    "a photo of an empty corridor or floor",
    "a photo of an exhibition booth, poster or signboard",
    "a blurry photo with no clear subject",
]

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(root: Path, limit=None):
    files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in VALID_EXTS]
    return files[:limit] if limit else files


def load_model(model_name: str):
    """CLIPモデルとプロセッサを読み込む"""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def make_scorer(model, processor, device):
    """
    画像バッチ(PIL.Imageのリスト)を受け取り、
    「コスプレらしさスコア(0〜1)」のリストを返す関数を作る。
    スコア = 肯定プロンプト群に割り当てられた確率の合計。
    """
    import torch

    prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
    n_pos = len(POSITIVE_PROMPTS)

    # テキスト側の特徴は一度だけ計算して使い回す
    with torch.no_grad():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        text_feat = model.get_text_features(**text_inputs)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    def score_batch(pil_images):
        with torch.no_grad():
            img_inputs = processor(images=pil_images, return_tensors="pt").to(device)
            img_feat = model.get_image_features(**img_inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = model.logit_scale.exp() * img_feat @ text_feat.T
            probs = logits.softmax(dim=-1)
            return probs[:, :n_pos].sum(dim=-1).cpu().tolist()

    return score_batch


def classify(score: float) -> str:
    if score >= THRESHOLD_HIGH:
        return "target"
    if score >= THRESHOLD_LOW:
        return "review"
    return "rejected"


def run(score_batch, src_root: Path, dst_root: Path):
    from PIL import Image

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw):
            return it

    images = list_images(src_root, LIMIT)
    print(f"対象画像: {len(images)} 枚（{src_root}）")
    if not images:
        print("画像が見つかりませんでした。SRC_ROOT を確認してください。")
        return

    dst_root.mkdir(parents=True, exist_ok=True)
    csv_path = dst_root / "scores.csv"
    counts = {"target": 0, "review": 0, "rejected": 0, "unreadable": 0}

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "score", "decision"])

        for i in tqdm(range(0, len(images), BATCH_SIZE), desc="CLIP判定"):
            batch_paths = images[i:i + BATCH_SIZE]
            pil_images, ok_paths = [], []
            for p in batch_paths:
                try:
                    with Image.open(p) as im:
                        pil_images.append(im.convert("RGB").copy())
                    ok_paths.append(p)
                except Exception:
                    rel = p.relative_to(src_root)
                    writer.writerow([str(rel), "", "unreadable"])
                    counts["unreadable"] += 1

            if not pil_images:
                continue

            scores = score_batch(pil_images)
            for p, s in zip(ok_paths, scores):
                rel = p.relative_to(src_root)
                decision = classify(s)
                counts[decision] += 1
                writer.writerow([str(rel), round(s, 4), decision])

                if DRY_RUN:
                    continue
                if decision == "rejected" and not COPY_REJECTED:
                    continue
                dst = dst_root / decision / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
            f.flush()  # 途中で止めても CSV が残るように

    total = sum(counts.values())
    print("\n" + "=" * 50)
    print("処理結果サマリ")
    print("=" * 50)
    for k in ["target", "review", "rejected", "unreadable"]:
        pct = counts[k] / total * 100 if total else 0
        print(f"  {k:10s}: {counts[k]:6d} 枚 ({pct:5.1f}%)")
    print(f"\nスコア一覧: {csv_path}")
    if DRY_RUN:
        print("※ DRY_RUN = True のため、画像のコピーは行っていません。")


def main():
    src_root = Path(SRC_ROOT)
    dst_root = Path(DST_ROOT)
    if not src_root.exists():
        print(f"エラー: 元フォルダが見つかりません: {src_root}")
        sys.exit(1)

    try:
        model, processor, device = load_model(MODEL_NAME)
    except ImportError:
        print("エラー: torch / transformers が見つかりません。")
        print("  pip install torch torchvision transformers pillow tqdm")
        sys.exit(1)

    score_batch = make_scorer(model, processor, device)
    run(score_batch, src_root, dst_root)


if __name__ == "__main__":
    main()