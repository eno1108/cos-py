import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import imagehash

# ==================== 設定エリア ====================
# 単一の動画ファイル、または動画が入っているフォルダのどちらを指定してもOK
TARGET_PATH = "/media/21a5026/Z5 1TB/iPhone/IMG_0066.MOV"

# 切り出したJPGの保存先
OUTPUT_DIR = "/media/21a5026/Z5 1TB/dataset_from_video"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 処理設定
FRAME_INTERVAL = 15      # 15フレーム（約0.5秒）おきに検査
SKIN_RATIO_MIN = 0.005   # 人物（肌色成分）が画面の0.5%以上あるか
BLUR_THRESHOLD = 25.0    # 手ブレ足切り値（少しのボケは許容）
DUP_THRESHOLD = 8        # 重複カット防止（pHash差分）
# ====================================================

VALID_VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.m4v')

def is_blurry(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD

def has_person(cv_img):
    """肌色検出による人物判定（多少の白飛び・暗さもカバー）"""
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    lower_skin1 = np.array([0, 25, 50], dtype=np.uint8)
    upper_skin1 = np.array([25, 200, 255], dtype=np.uint8)
    lower_skin2 = np.array([160, 25, 50], dtype=np.uint8)
    upper_skin2 = np.array([180, 200, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    skin_mask = mask1 | mask2

    ratio = np.sum(skin_mask > 0) / (cv_img.shape[0] * cv_img.shape[1])
    return ratio >= SKIN_RATIO_MIN

def enhance_image(pil_img):
    """画質補正（コントラストと輪郭を強調）"""
    enh_con = ImageEnhance.Contrast(pil_img)
    img_con = enh_con.enhance(1.1)
    enh_sharp = ImageEnhance.Sharpness(img_con)
    return enh_sharp.enhance(1.2)

def process_single_video(video_path, save_dir):
    v_name = os.path.splitext(os.path.basename(video_path))[0]
    sub_dir = os.path.join(save_dir, v_name)
    os.makedirs(sub_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"動画を開けませんでした: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n処理開始: {os.path.basename(video_path)} (総フレーム数: {total_frames})")

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

        # 1. 極端なブレを除外
        if is_blurry(frame):
            continue

        # 2. 人物（肌色成分）判定
        if not has_person(frame):
            continue

        # OpenCV(BGR) -> PIL(RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # 3. 重複カットの除外 (pHash)
        curr_hash = imagehash.phash(pil_img)
        is_dup = False
        for prev_hash in saved_hashes[-3:]:
            if curr_hash - prev_hash < DUP_THRESHOLD:
                is_dup = True
                break
        if is_dup:
            continue

        # 4. 軽い画質補正
        enhanced_img = enhance_image(pil_img)

        # 5. 高品質JPG保存
        out_name = f"{v_name}_f{frame_count:06d}.jpg"
        out_path = os.path.join(sub_dir, out_name)
        enhanced_img.save(out_path, "JPEG", quality=95)

        saved_hashes.append(curr_hash)
        saved_count += 1

        if saved_count % 10 == 0:
            print(f"抽出中: {saved_count} 枚保存済み... ({frame_count}/{total_frames})")

    cap.release()
    print(f"完了: {os.path.basename(video_path)} から {saved_count} 枚の人物JPGを抽出しました。")
    return saved_count

def main():
    if not os.path.exists(TARGET_PATH):
        print(f"エラー: パスが見つかりません: {TARGET_PATH}")
        return

    # ファイル単体かフォルダかを自動判別
    if os.path.isfile(TARGET_PATH):
        video_files = [TARGET_PATH]
    else:
        video_files = [
            os.path.join(TARGET_PATH, f) for f in os.listdir(TARGET_PATH)
            if f.lower().endswith(VALID_VIDEO_EXTS)
        ]

    if not video_files:
        print("処理対象の動画ファイルが見つかりませんでした。")
        return

    print(f"対象動画数: {len(video_files)} 本")
    total_saved = 0
    for v_path in video_files:
        total_saved += process_single_video(v_path, OUTPUT_DIR)

    print("\n==========================================")
    print(f"全処理が完了しました！ 合計: {total_saved} 枚のJPGを保存")
    print(f"保存先フォルダ: {OUTPUT_DIR}")
    print("==========================================")

if __name__ == "__main__":
    main()