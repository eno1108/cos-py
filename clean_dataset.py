import os
import shutil
import cv2
import imagehash
from PIL import Image

# ==================== 設定エリア ====================
# 写真が入っている元のフォルダ名
INPUT_DIR = "iPhone"  # または元の画像があるフォルダパスを指定

# 分類先の出力フォルダ
OUTPUT_DIR = "dataset_classified"
DIR_CLEAN = os.path.join(OUTPUT_DIR, "target_clean")      # ★顔が写っている綺麗なコスプレ候補
DIR_NOISE = os.path.join(OUTPUT_DIR, "venue_noise")       # 会場風景・ブース・後ろ姿（顔なし）
DIR_BLURRED = os.path.join(OUTPUT_DIR, "blurred")         # 手ブレ・ピンボケ
DIR_DUP = os.path.join(OUTPUT_DIR, "duplicates")          # 連写による重複カット

# 判定しきい値
BLUR_THRESHOLD = 80.0     # 手ブレ判定値（小さいほどボケている判定。80〜100程度が目安）
HASH_DIFF_THRESHOLD = 5   # 重複判定の類似度（小さいほど完全一致に近い）
# ====================================================

# 各出力フォルダを自動作成
for d in [DIR_CLEAN, DIR_NOISE, DIR_BLURRED, DIR_DUP]:
    os.makedirs(d, exist_ok=True)

# OpenCVの顔検出器（標準搭載のHaar Cascade）をロード
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 有効な画像拡張子
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.heic')

def is_blurred(image_cv):
    """Laplacian分散を用いて手ブレ・ピンボケを判定"""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < BLUR_THRESHOLD, variance

def has_face(image_cv):
    """顔が一定以上のサイズで写っているか判定"""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
    )
    return len(faces) > 0

def main():
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(VALID_EXTS)]
    image_files.sort()
    
    print(f"対象画像数: {len(image_files)} 枚の処理を開始します...")
    
    saved_hashes = []
    
    for idx, fname in enumerate(image_files):
        src_path = os.path.join(INPUT_DIR, fname)
        
        # 1. 画像の読み込み
        img_cv = cv2.imread(src_path)
        if img_cv is None:
            continue
            
        # 2. 手ブレ判定（Laplacian）
        blurred, score = is_blurred(img_cv)
        if blurred:
            shutil.copy2(src_path, os.path.join(DIR_BLURRED, fname))
            continue
            
        # 3. 顔検出（OpenCV）
        if not has_face(img_cv):
            # 顔がない ＝ 会場風景、看板、後ろ姿など
            shutil.copy2(src_path, os.path.join(DIR_NOISE, fname))
            continue
            
        # 4. 連写重複判定（pHash）
        try:
            pil_img = Image.open(src_path)
            current_hash = imagehash.phash(pil_img)
            
            # 直前に保存した綺麗画像と似すぎていないか比較
            is_duplicate = False
            for prev_hash in saved_hashes[-5:]:  # 直近5枚と比較
                if current_hash - prev_hash < HASH_DIFF_THRESHOLD:
                    is_duplicate = True
                    break
                    
            if is_duplicate:
                shutil.copy2(src_path, os.path.join(DIR_DUP, fname))
                continue
                
            # 全てのフィルタを通過：本命フォルダへコピー
            shutil.copy2(src_path, os.path.join(DIR_CLEAN, fname))
            saved_hashes.append(current_hash)
            
        except Exception as e:
            continue
            
        if (idx + 1) % 100 == 0:
            print(f"進捗: {idx + 1} / {len(image_files)} 完了")

    print("\n【分類完了】")
    print(f"- target_clean (本命コスプレ候補): {len(os.listdir(DIR_CLEAN))} 枚")
    print(f"- venue_noise  (会場風景・顔なし): {len(os.listdir(DIR_NOISE))} 枚")
    print(f"- blurred      (手ブレ・ボケ):     {len(os.listdir(DIR_BLURRED))} 枚")
    print(f"- duplicates   (連写重複):         {len(os.listdir(DIR_DUP))} 枚")

if __name__ == "__main__":
    main()
