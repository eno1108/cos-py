import os
import cv2
import numpy as np
from PIL import Image
import imagehash

# ==================== 設定エリア ====================
VIDEO_PATH = r"F:\iPhone\IMG_1307.MOV"  # 動画パス
OUTPUT_DIR = r"F:\dataset_from_video\IMG_1307_strict_cosplayers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAME_INTERVAL = 15      # 15フレームおき
BLUR_THRESHOLD = 25.0    # 手ブレ足切り
DUP_THRESHOLD = 10       # 重複判定（より厳しく）

# ★厳格判定パラメータ
VIVID_RATIO_MIN = 0.08   # 鮮やかな衣装・ウィッグ色が画面の 8% 以上（看板・通行人を排除）
SKIN_RATIO_MIN  = 0.03   # 肌色が画面の 3% 以上（遠くの通行人を排除）
# ====================================================

def is_blurry(gray_img):
    return cv2.Laplacian(gray_img, cv2.CV_64F).var() < BLUR_THRESHOLD

def is_strict_cosplayer(cv_img):
    h, w = cv_img.shape[:2]
    
    # 高速化のため縮小
    scale = 320.0 / w if w > 320 else 1.0
    sw, sh = int(w * scale), int(h * scale)
    small_bgr = cv2.resize(cv_img, (sw, sh))
    total_pixels = sw * sh

    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # 1. 鮮やかな衣装・ウィッグ（高彩度・明度）
    vivid_mask = (s >= 90) & (v >= 70)
    vivid_ratio = np.sum(vivid_mask) / total_pixels
    if vivid_ratio < VIVID_RATIO_MIN:
        return False

    # 2. まとまった肌色（顔や手足がしっかり写っているか）
    lower1 = np.array([0, 30, 60], dtype=np.uint8)
    upper1 = np.array([25, 200, 255], dtype=np.uint8)
    lower2 = np.array([160, 30, 60], dtype=np.uint8)
    upper2 = np.array([180, 200, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    skin_ratio = np.sum(skin_mask) / total_pixels
    if skin_ratio < SKIN_RATIO_MIN:
        return False

    # 3. 画面の中央エリア（メイン被写体）に肌色・衣装が存在するか
    # （画面端の見切れ通行人や、天井・床のノイズを完全にカット）
    center_y1, center_y2 = int(sh * 0.25), int(sh * 0.85)
    center_x1, center_x2 = int(sw * 0.20), int(sw * 0.80)
    center_skin = skin_mask[center_y1:center_y2, center_x1:center_x2]
    center_vivid = vivid_mask[center_y1:center_y2, center_x1:center_x2]

    center_pixels = (center_y2 - center_y1) * (center_x2 - center_x1)
    if (np.sum(center_skin) / center_pixels) < 0.025:
        return False
    if (np.sum(center_vivid) / center_pixels) < 0.05:
        return False

    return True

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"動画を開けませんでした: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"厳格判定で処理開始: {os.path.basename(video_path)} (総フレーム数: {total_frames})")

    frame_count = 0
    saved_count = 0
    saved_hashes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_INTERVAL != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 手ブレ判定
        if is_blurry(gray):
            continue

        # 厳格レイヤー判定（通路・看板・通行人を強力排除）
        if not is_strict_cosplayer(frame):
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        curr_hash = imagehash.phash(pil_img)

        # 重複カット排除
        if any(curr_hash - prev < DUP_THRESHOLD for prev in saved_hashes[-3:]):
            continue

        out_name = f"target_cosplay_{frame_count:06d}.jpg"
        pil_img.save(os.path.join(OUTPUT_DIR, out_name), "JPEG", quality=95)

        saved_hashes.append(curr_hash)
        saved_count += 1

        print(f"★レイヤー検出: frame {frame_count} を保存 (計 {saved_count} 枚)")

    cap.release()
    print(f"\n==========================================")
    print(f"厳格抽出完了！ 合計: {saved_count} 枚のレイヤー限定画像を保存")
    print(f"保存先: {OUTPUT_DIR}")
    print("==========================================")

if __name__ == "__main__":
    process_video(VIDEO_PATH)