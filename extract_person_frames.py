import os
import cv2
import numpy as np
from PIL import Image
import imagehash

# ==================== 設定エリア ====================
# ドライブ直下のフォルダを指定
BASE_DIR = r"F:\cos py"  # またはスクリプトのあるフォルダ
VIDEO_PATH = r"F:\iPhone\IMG_9565.MOV"  # または .mp4（フォルダ単位でも指定可能）
OUTPUT_DIR = r"F:\dataset_from_video\IMG_9565"  # 保存先フォルダ（存在しない場合は自動作成）
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 処理設定
FRAME_INTERVAL = 15      # 15フレーム（約0.5秒）おきに検査
BLUR_THRESHOLD = 25.0    # 手ブレ足切り値
DUP_THRESHOLD = 8        # 重複カット防止（pHash差分）
SKIN_RATIO_MIN = 0.015   # 人物判定：画面の1.5%以上に肌色成分があるか
# ====================================================

def is_blurry(gray_img):
    """Laplacian分散による手ブレ・ピンボケ判定"""
    return cv2.Laplacian(gray_img, cv2.CV_64F).var() < BLUR_THRESHOLD

def has_person(cv_img):
    """
    OpenCVの検出器クラスを使わず、HSV肌色領域と連結成分（塊の大きさ）で人物を判定。
    遠くの通行人や床・照明の誤認を弾き、コスプレイヤーやコンパニオンを抽出する。
    """
    h, w = cv_img.shape[:2]
    total_pixels = h * w

    # 高速化のため横幅640pxに縮小して検査
    scale = 640.0 / w if w > 640 else 1.0
    sw, sh = int(w * scale), int(h * scale)
    small_bgr = cv2.resize(cv_img, (sw, sh))

    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 25, 50], dtype=np.uint8)
    upper1 = np.array([25, 200, 255], dtype=np.uint8)
    lower2 = np.array([160, 25, 50], dtype=np.uint8)
    upper2 = np.array([180, 200, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    skin_ratio = np.sum(mask > 0) / (sw * sh)

    # 画面全体の肌色比率が下回っていれば即スキップ（通路や床を排除）
    if skin_ratio < SKIN_RATIO_MIN:
        return False

    # 肌色の「まとまった塊（顔や腕・上半身）」があるかをチェック（点在する遠景ノイズを弾く）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return False

    # 最大の肌色領域の面積が、縮小画面の0.5%以上あるか（遠くの豆粒通行人を排除）
    max_area = np.max(stats[1:, cv2.CC_STAT_AREA])
    return (max_area / (sw * sh)) >= 0.005

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

        # 1. 手ブレ判定
        if is_blurry(gray):
            continue

        # 2. 人物判定（通路・足元・遠景を弾く）
        if not has_person(frame):
            continue

        # 3. 重複カット判定（pHash）
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        curr_hash = imagehash.phash(pil_img)

        if any(curr_hash - prev < DUP_THRESHOLD for prev in saved_hashes[-3:]):
            continue

        # 4. JPG保存
        out_name = f"frame_{frame_count:06d}.jpg"
        pil_img.save(os.path.join(OUTPUT_DIR, out_name), "JPEG", quality=95)

        saved_hashes.append(curr_hash)
        saved_count += 1

        if saved_count % 10 == 0:
            print(f"抽出中: {saved_count} 枚保存済み...")

    cap.release()
    print(f"\n==========================================")
    print(f"抽出完了: {saved_count} 枚の人物JPG画像を保存しました。")
    print(f"保存先: {OUTPUT_DIR}")
    print("==========================================")

if __name__ == "__main__":
    process_video(VIDEO_PATH)