import os
import shutil
import numpy as np
from PIL import Image
import pillow_heif

# HEIC対応
pillow_heif.register_heif_opener()

# ==================== 設定エリア ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 元のiPhoneフォルダ（5,672枚）を直接指定
INPUT_DIR = os.path.join(BASE_DIR, "iPhone")

# 出力先フォルダ
DIR_TARGET = os.path.join(BASE_DIR, "dataset_final", "target_cosplay")  # ★本命（少しのボケ・白飛びOK）
DIR_VENUE = os.path.join(BASE_DIR, "dataset_final", "venue_noise")     # 会場風景・ブース（人なし）

# 肌色判定のしきい値（0.003＝全体の0.3%でも肌色があれば「人物あり」と判定して救済）
SKIN_RATIO_THRESHOLD = 0.003
# ====================================================

os.makedirs(DIR_TARGET, exist_ok=True)
os.makedirs(DIR_VENUE, exist_ok=True)

VALID_EXTS = ('.jpg', '.jpeg', '.png', '.heic')

def check_has_person(pil_img):
    """多少の白飛びや暗さがあっても、人物（肌色成分）を広めに拾う"""
    try:
        # リサイズして判定高速化
        img_small = pil_img.resize((150, 150)).convert('HSV')
        hsv_arr = np.array(img_small)

        h = hsv_arr[:, :, 0]
        s = hsv_arr[:, :, 1]
        v = hsv_arr[:, :, 2]

        # 白飛び気味の明るい肌色〜暗めの肌色まで幅広くカバー
        skin_mask = (
            ((h >= 0) & (h <= 30)) | ((h >= 235) & (h <= 255))
        ) & (s >= 25) & (s <= 200) & (v >= 40)

        skin_ratio = np.sum(skin_mask) / (150 * 150)
        return skin_ratio >= SKIN_RATIO_THRESHOLD
    except Exception:
        # 読み込みエラー等でも誤判定で捨てないよう本命側へ
        return True

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"エラー: フォルダ '{INPUT_DIR}' が見つかりません。")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(VALID_EXTS)]
    files.sort()
    total = len(files)
    print(f"全 {total} 枚の選別を開始します（少しのボケ・白ボケはすべて本命へ通します）...")

    target_count = 0
    venue_count = 0

    for idx, fname in enumerate(files):
        src_path = os.path.join(INPUT_DIR, fname)
        try:
            pil_img = Image.open(src_path)
        except Exception:
            continue

        if check_has_person(pil_img):
            # 人物が写っていれば（少しのピンボケ・白飛びもOK）本命へ
            shutil.copy2(src_path, os.path.join(DIR_TARGET, fname))
            target_count += 1
        else:
            # 天井・看板・完全な展示ブースのみ会場ノイズへ
            shutil.copy2(src_path, os.path.join(DIR_VENUE, fname))
            venue_count += 1

        if (idx + 1) % 200 == 0 or (idx + 1) == total:
            print(f"進捗: {idx + 1} / {total} 完了")

    print("\n【選別が完了しました！】")
    print(f"- target_cosplay (★少しのボケ・白飛びを含む本命コスプレ写真): {target_count} 枚")
    print(f"- venue_noise    (人なし会場風景・看板のみ):               {venue_count} 枚")

if __name__ == "__main__":
    main()
