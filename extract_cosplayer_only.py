import os
import cv2
import numpy as np
from PIL import Image
import imagehash

# ==================== 設定エリア ====================
VIDEO_PATH = r"F:\iPhone\IMG_1307.MOV"  # 今回の動画パス
OUTPUT_DIR = r"F:\dataset_from_video\IMG_1307_cosplayers_fixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAME_INTERVAL = 10      # 動きが速いため10フレームおきに短縮（約0.3秒間隔）
BLUR_THRESHOLD = 10.0    # ★ブレ判定を大幅に緩和（歩き撮りのブレを許容）
DUP_THRESHOLD = 6        # 重複判定
# ====================================================

def is_blurry(gray_img):
    """ブレが極端にひどいものだけ弾く（歩き撮りを救済）"""
    return cv2.Laplacian(gray_img, cv2.CV_64F).var() < BLUR_THRESHOLD

def is_cosplayer_rescued(cv_img):
    """
    衣装が黒・紺・制服系でも、
    赤・ピンク・紫・金・青などの特徴的なウィッグ＋肌色があれば拾う判定
    """
    h, w = cv_img.shape[:2]
    scale = 320.0 / w if w > 320 else 1.0
    sw, sh = int(w * scale), int(h * scale)
    small = cv2.resize(cv_img, (sw, sh))
    total_pixels = sw * sh

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0]
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    # 1. 肌色チェック（顔や手があるか）
    lower1 = np.array([0, 25, 50], dtype=np.uint8)
    upper1 = np.array([25, 200, 255], dtype=np.uint8)
    lower2 = np.array([165, 25, 50], dtype=np.uint8)
    upper2 = np.array([180, 200, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    skin_ratio = np.sum(skin_mask > 0) / total_pixels

    # 肌色が画面の 0.6% 以上（顔の一部でも写っていればOK）
    if skin_ratio < 0.006:
        return False

    # 2. ウィッグ・髪色チェック（赤・ピンク・紫・金・水色・緑など）
    # 赤〜ピンク系ウィッグ (H: 160-180 または 0-15 で高彩度)
    red_wig = (((h_chan >= 160) & (h_chan <= 180)) | ((h_chan >= 0) & (h_chan <= 12))) & (s_chan >= 70) & (v_chan >= 50)
    # 金髪・黄色系ウィッグ (H: 20-35)
    yellow_wig = (h_chan >= 20) & (h_chan <= 35) & (s_chan >= 70) & (v_chan >= 70)
    # 青・水色・紫系ウィッグ (H: 90-155)
    cool_wig = (h_chan >= 90) & (h_chan <= 155) & (s_chan >= 60) & (v_chan >= 50)

    wig_mask = red_wig | yellow_wig | cool_wig
    wig_ratio = np.sum(wig_mask) / total_pixels

    # ウィッグまたは高彩度衣装が画面の 1.2% 以上あれば合格
    return wig_ratio >= 0.012

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"動画を開けませんでした: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"処理開始: {os.path.basename(video_path)} (総フレーム: {total_frames})")

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

        # ブレ判定（緩め）
        if is_blurry(gray):
            continue

        # ウィッグ・人物判定
        if not is_cosplayer_rescued(frame):
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        curr_hash = imagehash.phash(pil_img)

        if any(curr_hash - prev < DUP_THRESHOLD for prev in saved_hashes[-3:]):
            continue

        out_name = f"target_{frame_count:06d}.jpg"
        pil_img.save(os.path.join(OUTPUT_DIR, out_name), "JPEG", quality=95)

        saved_hashes.append(curr_hash)
        saved_count += 1
        print(f"★フレーム保存: frame {frame_count} (計 {saved_count} 枚)")

    cap.release()
    print(f"\n完了！ {saved_count} 枚を保存しました: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_video(VIDEO_PATH)