import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import imagehash
from ultralytics import YOLO

#入力動画パス

TARGET_PATH = "F:\iPhone\🌟盾の勇者の成り上がり🐤フィーロ★コミックマーケット96🌈エアコミケ ❣.mp4"

OUTPUT_DIR = r"F:\dataset_from_video"

# 判定モード:
#   "cosplay" : 人物 + ウィッグ色（赤/ピンク/金/青/紫/白髪など）
#   "person"  : 人物が一定サイズ写っていれば全保存（まずは確実に拾いたい場合）
MODE = "cosplay"

# 処理パラメータ
FRAME_INTERVAL    = 5      # 5フレームおきに検査（約0.15秒ごと）
BLUR_THRESHOLD    = 4.0   # ★歩き撮り救済：30.0 -> 4.0 に緩和（動きブレを許容）
DUP_THRESHOLD     = 8      # 重複カット防止（pHash差分）
ENHANCE           = False  # 研究データ用：無加工保存（推奨）

# YOLOv8 人物検出設定
YOLO_MODEL        = "yolov8n.pt"
PERSON_CONF       = 0.35   # 検出の閾値（0.5 -> 0.35 に少し緩和）
MIN_PERSON_HEIGHT = 0.25   # 人物の高さが画面縦幅の25%以上（遠くの通行人をカット）

# ウィッグ判定設定（頭部上部40%領域）
WIG_RATIO_MIN     = 0.025  # 0.06 -> 0.025（頭部の2.5%以上にウィッグ色があれば合格）

SAVE_CROP         = False  # True: 人物の切り抜き画像も同時保存
DEBUG             = True   # True: 落選理由をターミナルに表示
# ====================================================



def blur_score(bgr_img):
    """Laplacian分散（大きいほど鮮明）"""
    if bgr_img is None or bgr_img.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def has_wig_color(person_bgr):
    """
    人物の頭部（上部40%）に、赤・ピンク・金・水色・紫・白髪などの
    特徴的なウィッグ色が含まれているかを判定
    """
    ph, pw = person_bgr.shape[:2]
    if ph < 10 or pw < 10:
        return False

    # 頭部領域の抽出
    head = person_bgr[: max(1, int(ph * 0.40)), :]
    hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 赤・ピンク系（有馬かな、赤髪、ピンク髪）
    red_wig = (((h >= 160) & (h <= 180)) | (h <= 10)) & (s >= 65) & (v >= 45)
    # 金髪・オレンジ系
    yellow_wig = (h >= 12) & (h <= 38) & (s >= 60) & (v >= 60)
    # 青・水色・紫・緑系
    cool_wig = (h >= 40) & (h <= 155) & (s >= 45) & (v >= 45)
    # 白髪・銀髪系（低彩度・高明度）
    white_wig = (s <= 35) & (v >= 180)

    wig_mask = red_wig | yellow_wig | cool_wig | white_wig
    return wig_mask.mean() >= WIG_RATIO_MIN


def find_persons(result, frame_h, frame_w):
    """画面内から一定サイズ以上の人物bboxを大きい順に取得"""
    found = []
    for box in result.boxes:
        if int(box.cls[0]) != 0:  # 0 = person
            continue
        conf = float(box.conf[0])
        if conf < PERSON_CONF:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame_w), min(y2, frame_h)

        # 画面に対する高さ比率
        if (y2 - y1) / frame_h < MIN_PERSON_HEIGHT:
            continue

        found.append(((x2 - x1) * (y2 - y1), (x1, y1, x2, y2)))

    found.sort(reverse=True)
    return [b for _, b in found]


def pick_target(frame, bboxes):
    """ブレ判定とウィッグ判定を通過するターゲット人物を選定"""
    if not bboxes:
        return None, 0.0, "人物なし（遠すぎる/未検出）"

    reason = ""
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        person = frame[y1:y2, x1:x2]

        score = blur_score(person)
        if score < BLUR_THRESHOLD:
            reason = f"手ブレ (score={score:.1f} < {BLUR_THRESHOLD})"
            continue

        if MODE == "cosplay" and not has_wig_color(person):
            reason = "ウィッグ色なし（一般通行人または黒髪）"
            continue

        return bbox, score, ""

    return None, 0.0, reason


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
    print(f"\n処理開始: {os.path.basename(video_path)} (総フレーム数: {total_frames}, MODE: {MODE})")

    frame_count = 0
    saved_count = 0
    last = None  # (hash, path, score)

    while True:
        if not cap.grab():
            break

        frame_count += 1
        if frame_count % FRAME_INTERVAL != 0:
            continue

        ret, frame = cap.retrieve()
        if not ret:
            continue

        h, w = frame.shape[:2]

        # 1. 人物検出
        res = model.predict(frame, classes=[0], conf=PERSON_CONF, verbose=False)[0]
        bboxes = find_persons(res, h, w)

        # 2. ターゲット選定（ブレ・ウィッグ判定）
        bbox, score, reason = pick_target(frame, bboxes)
        if bbox is None:
            if DEBUG:
                print(f"  × frame {frame_count:06d}: {reason}")
            continue

        x1, y1, x2, y2 = bbox

        # 3. 重複判定（pHash）
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        curr_hash = imagehash.phash(pil_img)
        out_path = os.path.join(sub_dir, f"{v_name}_f{frame_count:06d}.jpg")

        if last and curr_hash - last[0] < DUP_THRESHOLD:
            # 似た構図の場合、よりブレの少ない（シャープな）方を残す
            if score > last[2]:
                if os.path.exists(last[1]):
                    os.remove(last[1])
                save_photo(pil_img, out_path)
                last = (curr_hash, out_path, score)
                print(f"  ★frame {frame_count:06d} を上書き更新（より鮮明: score={score:.1f}）")
            continue

        # 4. 保存
        save_photo(pil_img, out_path)

        if SAVE_CROP:
            pad = int(0.05 * max(x2 - x1, y2 - y1))
            c = frame[max(y1 - pad, 0):min(y2 + pad, h), max(x1 - pad, 0):min(x2 + pad, w)]
            Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)).save(
                out_path.replace(".jpg", "_crop.jpg"), "JPEG", quality=95
            )

        last = (curr_hash, out_path, score)
        saved_count += 1
        print(f"★保存成功: frame {frame_count:06d} (score={score:.1f}, 計 {saved_count} 枚)")

    cap.release()
    print(f"\n抽出完了: {saved_count} 枚を保存しました -> {sub_dir}")
    return saved_count


def main():
    if not os.path.exists(TARGET_PATH):
        print(f"エラー: 指定されたパスが見つかりません: {TARGET_PATH}")
        return

    video_files = [TARGET_PATH] if os.path.isfile(TARGET_PATH) else sorted(
        [os.path.join(TARGET_PATH, f) for f in os.listdir(TARGET_PATH) if f.lower().endswith(VALID_VIDEO_EXTS)]
    )

    if not video_files:
        print("動画ファイルが見つかりませんでした。")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(YOLO_MODEL)

    total = sum(process_single_video(vp, OUTPUT_DIR, model) for vp in video_files)
    print(f"\n==========================================")
    print(f"全動画処理完了！ 合計: {total} 枚のJPGを保存")
    print(f"保存先: {OUTPUT_DIR}")
    print(f"==========================================")

# --- main() の中にインデントを揃えて配置 ---
    import ctypes
    import winsound

    # 完了音を鳴らす
    winsound.MessageBeep(winsound.MB_ICONASTERISK)
    # 画面に完了メッセージを表示
    ctypes.windll.user32.MessageBoxW(0, f"動画からの画像抽出が完了しました！\n合計: {total} 枚", "処理完了", 0x40)


if __name__ == "__main__":
    main()
