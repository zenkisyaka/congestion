import torch
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# YOLOモデルをロード
model = YOLO("/Users/chinenyoshinori/congestion-1/runs/detect/train6/weights/best.pt")  # モデルを指定

# 推論画像のパス
image_path = "/Users/chinenyoshinori/congestion-1/data/images/image_7.jpg"

# 推論を実行
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
target_class = 0
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

