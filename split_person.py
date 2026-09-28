import os
import shutil
import numpy as np
from PIL import Image
import pillow_heif

# HEIC形式のデコードを有効化
pillow_heif.register_heif_opener()

# ==================== 設定エリア ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 先ほど選別された2,924枚が入っているフォルダ
INPUT_DIR = os.path.join(BASE_DIR, "dataset_classified", "target_clean")

# 出力先フォルダ
DIR_PERSON = os.path.join(BASE_DIR, "dataset_classified", "person_detected")      # ★人物あり（コスプレ対象）
DIR_NO_PERSON = os.path.join(BASE_DIR, "dataset_classified", "no_person_venue")   # 会場風景・ブース・展示のみ

# 人物判定の肌色ピクセル比率（画面全体の0.8%以上に肌色成分があるか）
SKIN_RATIO_THRESHOLD = 0.008
# ====================================================

os.makedirs(DIR_PERSON, exist_ok=True)
os.makedirs(DIR_NO_PERSON, exist_ok=True)

VALID_EXTS = ('.jpg', '.jpeg', '.png', '.heic')

def contains_person(pil_img):
    """HSV色空間における肌色ピクセルの割合から人物の有無を判定"""
    try:
        # 判定高速化のため低解像度にリサイズ
        img_small = pil_img.resize((200, 200)).convert('HSV')
        hsv_arr = np.array(img_small)

        h = hsv_arr[:, :, 0]  # 色相 (0〜255)
        s = hsv_arr[:, :, 1]  # 彩度 (0〜255)
        v = hsv_arr[:, :, 2]  # 明度 (0〜255)

        # 肌色のHSV範囲条件
        skin_mask = (
            ((h >= 0) & (h <= 25)) | ((h >= 240) & (h <= 255))
        ) & (s >= 40) & (s <= 180) & (v >= 60)

        skin_ratio = np.sum(skin_mask) / (200 * 200)
        return skin_ratio >= SKIN_RATIO_THRESHOLD, skin_ratio
    except Exception:
        return True, 1.0  # エラー時は安全側に倒して人物フォルダへ

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"エラー: フォルダ '{INPUT_DIR}' が見つかりません。")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(VALID_EXTS)]
    files.sort()
    total = len(files)
    print(f"対象画像数: {total} 枚の人物判定を開始します...")

    person_count = 0
    no_person_count = 0

    for idx, fname in enumerate(files):
        src_path = os.path.join(INPUT_DIR, fname)
        try:
            pil_img = Image.open(src_path)
        except Exception:
            continue

        has_person, ratio = contains_person(pil_img)

        if has_person:
            shutil.copy2(src_path, os.path.join(DIR_PERSON, fname))
            person_count += 1
        else:
            shutil.copy2(src_path, os.path.join(DIR_NO_PERSON, fname))
            no_person_count += 1

        if (idx + 1) % 200 == 0 or (idx + 1) == total:
            print(f"進捗: {idx + 1} / {total} 完了")

    print("\n【人物分類が完了しました】")
    print(f"- person_detected (★人が写っている画像): {person_count} 枚")
    print(f"- no_person_venue (風景・ブース展示のみ): {no_person_count} 枚")

if __name__ == "__main__":
    main()
