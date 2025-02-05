import torch
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# YOLOモデルをロード
model = YOLO("/Users/chinenyoshinori/congestion-1/runs/detect/train/weights/best.pt")

# 推論画像のパス
image_path = "/Users/chinenyoshinori/congestion-1/data/images/LINE_ALBUM_YoLo用画像_250205_3.jpg"
results = model(image_path)

# 推論結果を取得
predictions = results[0].boxes  

# 確率とバウンディングボックス情報を取得
scores = predictions.conf.cpu().numpy()  # GPU対応
classes = predictions.cls.cpu().numpy()  
bboxes = predictions.xyxy.cpu().numpy()  


def plot_probability_map(image_path, bboxes, scores, classes, target_class):
    # 画像を読み込み
    image = plt.imread(image_path)

    # 画像のサイズ取得
    height, width, _ = image.shape

    # 空の確率マップを作成
    prob_map = np.zeros((height, width))

    # ターゲットクラスの確率をマッピング
    for bbox, score, cls in zip(bboxes, scores, classes):
        if int(cls) == target_class:  # クラスIDを整数に変換して比較
            x_min, y_min, x_max, y_max = map(int, bbox)

            # 画像の範囲内に収める
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(width, x_max), min(height, y_max)

            prob_map[y_min:y_max, x_min:x_max] += score

    # 確率マップを正規化（最大値が0でない場合のみ）
    if prob_map.max() > 0:
        prob_map = prob_map / prob_map.max()

    # 可視化
    plt.figure(figsize=(10, 10))
    plt.imshow(image, alpha=0.8)  # 元画像
    plt.imshow(prob_map, cmap='jet', alpha=0.5)  # 確率マップを重ねる
    plt.colorbar(label="Probability")
    plt.axis("off")
    plt.title(f"Probability Map for Class {target_class}")
    plt.show()


# ターゲットクラスを指定（例: 0 = 人, 1 = 車 など）
target_class = 0
plot_probability_map(image_path, bboxes, scores, classes, target_class)
