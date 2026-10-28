import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from PIL import Image
from ultralytics import YOLO
import imagehash


# ==================== 設定エリア ====================
# 単一の動画ファイル、または動画が入っているフォルダのどちらでもOK
#TARGET_PATH = "/media/21a5026/Z5 1TB/iPhone/IMG_2517.MOV"
#OUTPUT_DIR = "/media/21a5026/Z5 1TB/dataset_from_video"
TARGET_PATH = r"F:\iPhone\IMG_3128.MOV"
OUTPUT_DIR = r"F:\dataset_from_video"

# 判定モード
#   "person"  : 人物が大きく写っていればOK（黒い衣装・制服系も拾う）
#   "cosplay" : 人物 + 頭部にウィッグ色（赤/ピンク/金/青/紫など）がある（通行人を強く排除）
#MODE = "person"
MODE = "cosplay"

# 処理設定
FRAME_INTERVAL = 1       # 5フレームおきに検査（約0.15秒）
BLUR_THRESHOLD = 30.0     # ★人物部分のブレ足切り（歩き撮りで落ちすぎるなら 20 などに下げる）
DUP_THRESHOLD = 8         # 重複カット（pHash差分）
ENHANCE = False           # True: 軽い画質補正（コントラスト1.1 / シャープ1.2）

# 人物検出（YOLOv8）
YOLO_MODEL = "yolov8n.pt"  # 精度重視なら yolov8s.pt / yolov8m.pt
PERSON_CONF = 0.5          # 人物の信頼度
MIN_PERSON_HEIGHT = 0.35   # 人物の高さが画面の35%以上（遠くの通行人を排除）

# MODE="cosplay" 用：頭部（人物bboxの上部40%）に占めるウィッグ色の割合
WIG_RATIO_MIN = 0.06

SAVE_CROP = False          # True: 人物クロップ画像も別に保存（LoRA学習向け）
DEBUG = False              # True: 保存されなかったフレームの理由を表示（調整用）
# ====================================================

VALID_VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.m4v')


def blur_score(bgr_img):
    """Laplacian分散（大きいほどシャープ）"""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def find_persons(result, frame_h, frame_w):
    """条件を満たす人物bboxを、大きい順にすべて返す"""
    found = []
    for box in result.boxes:
        if int(box.cls[0]) != 0:  # 0 = person
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame_w), min(y2, frame_h)
        if (y2 - y1) / frame_h < MIN_PERSON_HEIGHT:
            continue
        found.append(((x2 - x1) * (y2 - y1), (x1, y1, x2, y2)))
    found.sort(reverse=True)
    return [b for _, b in found]


def pick_target(frame, bboxes):
    """
    全員を調べて、ブレ判定（とcosplayモードならウィッグ判定）を通る
    最大の人物を返す。 (bbox, score, 落選理由) 。該当なしなら bbox=None
    """
    reason = "人物なし（小さい/未検出）"
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        person = frame[y1:y2, x1:x2]
        score = blur_score(person)
        if score < BLUR_THRESHOLD:
            reason = f"ブレ (score={score:.0f} < {BLUR_THRESHOLD:.0f})"
            continue
        if MODE == "cosplay" and not has_wig_color(person):
            reason = "ウィッグ色なし"
            continue
        return bbox, score, ""
    return None, 0.0, reason


def has_wig_color(person_bgr):
    """人物の頭部付近に、赤・ピンク・金・青・紫系などの特徴的な髪色があるか"""
    ph = person_bgr.shape[0]
    head = person_bgr[: max(1, int(ph * 0.4))]
    hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 赤〜ピンク系（肌色との誤判定を減らすため彩度を高めに）
    red_wig = (((h >= 160) & (h <= 180)) | (h <= 8)) & (s >= 90) & (v >= 50)
    # 金髪・黄色系
    yellow_wig = (h >= 20) & (h <= 35) & (s >= 70) & (v >= 70)
    # 青・水色・紫系
    cool_wig = (h >= 90) & (h <= 155) & (s >= 60) & (v >= 50)

    wig_mask = red_wig | yellow_wig | cool_wig
    return wig_mask.mean() >= WIG_RATIO_MIN


def enhance_image(pil_img):
    img = ImageEnhance.Contrast(pil_img).enhance(1.1)
    return ImageEnhance.Sharpness(img).enhance(1.2)


def save_photo(pil_img, path):
    if ENHANCE:
        pil_img = enhance_image(pil_img)
    pil_img.save(path, "JPEG", quality=95)


def process_single_video(video_path, save_dir, model):
    v_name = os.path.splitext(os.path.basename(video_path))[0]
    sub_dir = os.path.join(save_dir, v_name)
    os.makedirs(sub_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"動画を開けませんでした: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n処理開始: {os.path.basename(video_path)} (総フレーム数: {total_frames}, モード: {MODE})")

    frame_count = 0
    saved_count = 0
    last = None  # 直前に保存した写真 (hash, path, score)

    while True:
        # grab()だけで読み飛ばすと、不要フレームのデコードを省けて速い
        if not cap.grab():
            break
        frame_count += 1
        if frame_count % FRAME_INTERVAL != 0:
            continue
        ret, frame = cap.retrieve()
        if not ret:
            continue

        h, w = frame.shape[:2]

        # 1. 人物検出（遠景の通行人・看板・床を排除）
        res = model.predict(frame, classes=[0], conf=PERSON_CONF, verbose=False)[0]
        bboxes = find_persons(res, h, w)

        # 2. 全員を調べ、ブレ判定（+ cosplayモードならウィッグ判定）を通る人物を探す
        bbox, score, reason = pick_target(frame, bboxes)
        if bbox is None:
            if DEBUG:
                print(f"  × frame {frame_count}: {reason}")
            continue
        x1, y1, x2, y2 = bbox

        # 3. 重複判定（連続して似ていれば、よりシャープな方だけ残す）
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        curr_hash = imagehash.phash(pil_img)
        out_path = os.path.join(sub_dir, f"{v_name}_f{frame_count:06d}.jpg")

        if last and curr_hash - last[0] < DUP_THRESHOLD:
            if score > last[2]:
                if os.path.exists(last[1]):
                    os.remove(last[1])
                save_photo(pil_img, out_path)
                last = (curr_hash, out_path, score)
            continue

        # 4. 保存
        save_photo(pil_img, out_path)
        if SAVE_CROP:
            pad = int(0.05 * max(x2 - x1, y2 - y1))
            c = frame[max(y1 - pad, 0):min(y2 + pad, h), max(x1 - pad, 0):min(x2 + pad, w)]
            Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)).save(
                out_path.replace(".jpg", "_crop.jpg"), "JPEG", quality=95)

        last = (curr_hash, out_path, score)
        saved_count += 1
        print(f"★保存: frame {frame_count} (計 {saved_count} 枚)")

    cap.release()
    print(f"完了: {os.path.basename(video_path)} から {saved_count} 枚を抽出しました。")
    return saved_count


def main():
    if not os.path.exists(TARGET_PATH):
        print(f"エラー: パスが見つかりません: {TARGET_PATH}")
        return

    if os.path.isfile(TARGET_PATH):
        video_files = [TARGET_PATH]
    else:
        video_files = sorted(
            os.path.join(TARGET_PATH, f) for f in os.listdir(TARGET_PATH)
            if f.lower().endswith(VALID_VIDEO_EXTS)
        )

    if not video_files:
        print("処理対象の動画ファイルが見つかりませんでした。")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(YOLO_MODEL)  # 初回のみ自動ダウンロード

    print(f"対象動画数: {len(video_files)} 本")
    total_saved = 0
    for v_path in video_files:
        total_saved += process_single_video(v_path, OUTPUT_DIR, model)

    print("\n==========================================")
    print(f"全処理が完了しました！ 合計: {total_saved} 枚のJPGを保存")
    print(f"保存先フォルダ: {OUTPUT_DIR}")
    print("==========================================")


if __name__ == "__main__":
    main()