<<<<<<< HEAD

=======
>>>>>>> zenki3
import torch
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# YOLOモデルをロード
<<<<<<< HEAD
model = YOLO("/Users/chinenyoshinori/congestion-1/runs/detect/train16/weights/best.pt")  # モデルを指定

# 推論画像のパス
image_path = "/Users/chinenyoshinori/congestion-1/data/images/LINE_ALBUM_YoLo用画像_250205_3.jpg"
=======
model = YOLO("/Users/chinenyoshinori/congestion-1/runs/detect/train6/weights/best.pt")  # モデルを指定

# 推論画像のパス
image_path = "/Users/chinenyoshinori/congestion-1/data/images/image_7.jpg"

# 推論を実行
>>>>>>> zenki3
results = model(image_path)


# 推論結果を取得
predictions = results[0].boxes  # バウンディングボックス

# 確率とバウンディングボックス情報を取得
scores = predictions.conf.numpy()  # 確率スコア
classes = predictions.cls.numpy()  # クラスID
bboxes = predictions.xyxy.numpy()  # バウンディングボックス (x_min, y_min, x_max, y_max)


def plot_probability_map(image_path, bboxes, scores, classes, target_class):
    # 画像を読み込み
    image = plt.imread(image_path)

    # 画像のサイズ取得
    height, width, _ = image.shape

    # 空の確率マップを作成
    prob_map = np.zeros((height, width))

    # ターゲットクラスの確率をマッピング
    for bbox, score, cls in zip(bboxes, scores, classes):
        if cls == target_class:  # ターゲットクラスのみ
            x_min, y_min, x_max, y_max = map(int, bbox)
            prob_map[y_min:y_max, x_min:x_max] += score

    # 確率マップを正規化
    prob_map = prob_map / prob_map.max()

    # 可視化
    plt.figure(figsize=(10, 10))
    plt.imshow(image, alpha=0.8)
    plt.imshow(prob_map, cmap='jet', alpha=0.5)  # 確率マップを重ねる
    plt.colorbar(label="Probability")
    plt.axis("off")
    plt.title(f"Probability Map for Class {target_class}")
    plt.show()

# ターゲットクラスを指定 (例: 0 = person)
<<<<<<< HEAD
target_class = 2
=======
target_class = 0
>>>>>>> zenki3
plot_probability_map(image_path, bboxes, scores, classes, target_class)

import os
from PIL import Image

def count_images_in_folder(folder_path):
    image_count = 0

    # フォルダ内のファイルを取得
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # ファイルかどうか確認
        if os.path.isfile(file_path):
            try:
                # 画像として開ける場合、カウントする
                with Image.open(file_path) as img:
                    image_count += 1
            except Exception:
                # 画像として開けなければスキップ
                pass

    return image_count

<<<<<<< HEAD
import os
from PIL import Image

def get_image_sizes(directory):
    """
    指定したディレクトリ内の全ての画像ファイルのサイズを取得する。
    
    Args:
        directory (str): 画像が格納されているディレクトリのパス。
        
    Returns:
        list: 各画像のファイル名とサイズ (幅, 高さ) のリスト。
    """
    image_sizes = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
                    image_sizes.append((filename, width, height))
            except Exception as e:
                print(f"画像の読み込みに失敗しました: {filename}, エラー: {e}")
    return image_sizes


# ディレクトリのパスを指定
image_directory = "/Users/chinenyoshinori/congestion-1/data/images"

# 画像サイズを取得
sizes = get_image_sizes(image_directory)

# 結果を表示
print("画像サイズ:")
for filename, width, height in sizes:
    print(f"{filename}: {width}x{height}")


import os

def merge_annotation_files(detected_folder, annotated_folder, output_folder):
    """
    YOLOの推論結果と手動アノテーションのデータを結合し、新しいフォルダに保存する。

    Args:
        detected_folder (str): YOLOで検出されたバウンディングボックスのデータがあるフォルダ
        annotated_folder (str): 手動アノテーションデータがあるフォルダ
        output_folder (str): 結合後のアノテーションを保存するフォルダ
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # フォルダ内のすべてのテキストファイルを取得
    detected_files = {f for f in os.listdir(detected_folder) if f.endswith(".txt")}
    annotated_files = {f for f in os.listdir(annotated_folder) if f.endswith(".txt")}

    # すべてのアノテーションファイルのリスト
    all_files = detected_files | annotated_files

    for filename in all_files:
        merged_content = []

        # YOLOの推論データがある場合、追加
        if filename in detected_files:
            with open(os.path.join(detected_folder, filename), "r", encoding="utf-8") as f:
                merged_content.extend(f.readlines())

        # 手動アノテーションデータがある場合、追加
        if filename in annotated_files:
            with open(os.path.join(annotated_folder, filename), "r", encoding="utf-8") as f:
                merged_content.extend(f.readlines())

        # 結合したデータを新しいフォルダに保存
        output_path = os.path.join(output_folder, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(merged_content)

        print(f"結合完了: {output_path}")

# 入力フォルダ（YOLOの推論結果 & 手動アノテーション）
detected_folder = "/Users/chinenyoshinori/congestion-1/dataset/val/ano_labels"  
annotated_folder = "/Users/chinenyoshinori/congestion-1/dataset/val/YOLO_labels"

# 出力フォルダ（結合されたアノテーション）
output_folder = "/Users/chinenyoshinori/congestion-1/dataset2/val/labels"

# 実行
merge_annotation_files(detected_folder, annotated_folder, output_folder)


import os
import shutil

def copy_files_keep_original(source_folder, destination_folder):
    """
    指定したフォルダ内のすべてのファイルを、フォルダ構造を保ったまま別のフォルダにコピーする。
    （元のファイルとフォルダはそのまま残す）

    Args:
        source_folder (str): コピー元のフォルダ
        destination_folder (str): コピー先のフォルダ
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for root, dirs, files in os.walk(source_folder):
        # コピー先のフォルダパスを作成
        relative_path = os.path.relpath(root, source_folder)
        destination_path = os.path.join(destination_folder, relative_path)

        # フォルダをコピー先に作成（元の構造を維持）
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        for file_name in files:
            source_file = os.path.join(root, file_name)
            destination_file = os.path.join(destination_path, file_name)

            # ファイルをコピー
            shutil.copy2(source_file, destination_file)
            print(f"コピー完了: {source_file} → {destination_file}")

# 元のフォルダとコピー先のフォルダを指定
source_folder = "/Users/chinenyoshinori/congestion-1/dataset/val/images"
destination_folder = "/Users/chinenyoshinori/congestion-1/dataset2/val/images"

# 実行
copy_files_keep_original(source_folder, destination_folder)

from ultralytics import YOLO

# 事前学習済みモデル（yolov8m.pt）をロード
model = YOLO("yolov8m.pt")

# CPUで再学習の実行
model.train(data="dataset2/data.yaml", epochs=50, batch=8, imgsz=640, device="cpu")

import os

def fix_labels(label_folder):
    """
    ラベルフォルダ内のYOLOアノテーション(.txt)の座標値を正規化（0〜1の範囲に修正）
    
    Args:
        label_folder (str): ラベルが格納されているフォルダのパス
    """
    for label_file in os.listdir(label_folder):
        if label_file.endswith(".txt"):
            file_path = os.path.join(label_folder, label_file)

            with open(file_path, "r") as f:
                lines = f.readlines()

            fixed_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:  # 正しい形式
                    class_id = parts[0]
                    coords = [float(x) for x in parts[1:]]  # x, y, w, h
                    # すべての座標値を0-1の範囲にスケール
                    coords = [max(0, min(1, x)) for x in coords]
                    fixed_lines.append(f"{class_id} {' '.join(map(str, coords))}\n")

            # 修正後のラベルを保存
            with open(file_path, "w") as f:
                f.writelines(fixed_lines)

            print(f"Fixed: {file_path}")

# ラベルフォルダのパス（修正するフォルダを指定）
train_label_folder = "/Users/chinenyoshinori/congestion-1/dataset2/train/labels"
val_label_folder = "/Users/chinenyoshinori/congestion-1/dataset2/val/labels"

# 修正を実行
fix_labels(train_label_folder)
fix_labels(val_label_folder)

import os

def fix_labels(label_folder):
    """
    YOLOアノテーション (.txt) の座標値を 0〜1 の範囲に修正するスクリプト

    Args:
        label_folder (str): ラベルが格納されているフォルダのパス
    """
    for label_file in os.listdir(label_folder):
        if label_file.endswith(".txt"):
            file_path = os.path.join(label_folder, label_file)

            with open(file_path, "r") as f:
                lines = f.readlines()

            fixed_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:  # YOLOフォーマット (クラスID, x, y, w, h)
                    class_id = parts[0]
                    coords = [float(x) for x in parts[1:]]  # x, y, w, h
                    # 0〜1の範囲にスケール
                    coords = [max(0, min(1, x)) for x in coords]
                    fixed_lines.append(f"{class_id} {' '.join(map(str, coords))}\n")

            # 修正後のラベルを保存
            with open(file_path, "w") as f:
                f.writelines(fixed_lines)

            print(f"Fixed: {file_path}")

# ラベルフォルダのパス
train_label_folder = "/Users/chinenyoshinori/congestion-1/dataset2/train/labels"
val_label_folder = "/Users/chinenyoshinori/congestion-1/dataset2/val/labels"

# 修正を実行
fix_labels(train_label_folder)
fix_labels(val_label_folder)



from ultralytics import YOLO

# YOLOv8mで学習を開始
model = YOLO("yolov8m.pt")
model.train(data="dataset2/data.yaml", epochs=50, batch=8, imgsz=640, device="cpu")


import os

# 2つのフォルダのパス
folder1 = "/Users/chinenyoshinori/congestion-1/dataset/val/ano_labels"  # 例: "/Users/user/folder1"
folder2 = "/Users/chinenyoshinori/congestion-1/dataset/val/labels"  # 例: "/Users/user/folder2"
merged_folder = "ano_labels2"  # マージ後のファイルを保存するフォルダ

# マージフォルダがなければ作成
os.makedirs(merged_folder, exist_ok=True)

# folder1 のファイル一覧を取得
files1 = set(os.listdir(folder1))
files2 = set(os.listdir(folder2))

# 両方のフォルダに存在するテキストファイルを探す
common_files = files1 & files2  # 共通のファイル名のみ
common_txt_files = {f for f in common_files if f.endswith(".txt")}

# ファイルをマージ
for filename in common_txt_files:
    file1_path = os.path.join(folder1, filename)
    file2_path = os.path.join(folder2, filename)
    merged_file_path = os.path.join(merged_folder, filename)

    # ファイルの内容を読み込む
    with open(file1_path, "r", encoding="utf-8") as f1, open(file2_path, "r", encoding="utf-8") as f2:
        content1 = f1.read()
        content2 = f2.read()

    # マージした内容を書き込む
    with open(merged_file_path, "w", encoding="utf-8") as mf:
        mf.write(content1 + "\n" + content2)

    print(f"✅ {filename} をマージしました！ → {merged_file_path}")

print("\n🎉 すべてのファイルのマージが完了しました！")

import os
import shutil

# 元のフォルダとコピー先のフォルダを指定
source_folder = "/Users/chinenyoshinori/congestion-1/dataset/val/ano_labels"  # コピー元
destination_folder = "/Users/chinenyoshinori/congestion-1/dataset2/val/labels"  # コピー先

# コピー先フォルダがなければ作成
os.makedirs(destination_folder, exist_ok=True)

# 元のフォルダ内のファイルをコピー
for filename in os.listdir(source_folder):
    source_path = os.path.join(source_folder, filename)
    destination_path = os.path.join(destination_folder, filename)

    # ファイルならコピー（フォルダはスキップ）
    if os.path.isfile(source_path):
        shutil.copy2(source_path, destination_path)
        print(f"✅ {filename} を {destination_folder} にコピーしました")

print("\n🎉 すべてのファイルをコピーしました！")

model.train(
    data="dataset2/data.yaml",
    epochs=50,
    batch=8,
    imgsz=1920, #最大サイズ
    rect=True,   # 変換しない
    device="cpu"
)


import torch
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# モデルをロード
model = YOLO("/Users/chinenyoshinori/congestion-1/runs/detect/train6/weights/best.pt")

# 推論画像のパス
image_path = "/Users/chinenyoshinori/congestion-1/data/backup/detection/image_8.jpg"

# 推論の実行
results = model(image_path)

# 予測結果を取得
predictions = results[0].boxes
scores = predictions.conf.cpu().numpy()  # GPUの場合はCPUへ移動
classes = predictions.cls.cpu().numpy()
bboxes = predictions.xyxy.cpu().numpy()

def plot_probability_map(image_path, bboxes, scores, classes, target_class):
    """確率マップをプロットする"""
    # 画像を読み込み
    image = plt.imread(image_path)
    
    # 画像のサイズ取得
    height, width, _ = image.shape
    
    # 空の確率マップを作成
    prob_map = np.zeros((height, width))

    # ターゲットクラスの確率をマッピング
    for bbox, score, cls in zip(bboxes, scores, classes):
        if int(cls) == target_class:  # ターゲットクラスのみ処理
            x_min, y_min, x_max, y_max = map(int, bbox)
            prob_map[y_min:y_max, x_min:x_max] += score  # スコアを加算

    # 確率マップを正規化
    if prob_map.max() > 0:
        prob_map = prob_map / prob_map.max()

    # 可視化
    plt.figure(figsize=(10, 10))
    plt.imshow(image, alpha=0.8)  # 元の画像
    plt.imshow(prob_map, cmap='jet', alpha=0.5)  # 確率マップを重ねる
    plt.colorbar(label="Probability")
    plt.axis("off")
    plt.title(f"Probability Map for Class {target_class}")
    plt.show()

# ターゲットクラス（例: クラスID 2）
target_class = 2

# 確率マップをプロット
plot_probability_map(image_path, bboxes, scores, classes, target_class)

from ultralytics import YOLO

# YOLOv8モデルのロード
model = YOLO("yolov8m.pt")  # 事前学習済みモデル

model.train(
    data="dataset2/data.yaml",
    epochs=50,
    batch=8,
    imgsz=1920,  # 最大画像サイズを指定
    multi_scale=True,  # 可変サイズで学習
    device="cuda"
)

=======
>>>>>>> zenki3
