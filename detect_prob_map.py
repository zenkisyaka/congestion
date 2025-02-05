import torch
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# モデルをロード
model = YOLO("/Users/chinenyoshinori/congestion-1/runs/detect/train5/weights/best.pt")

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

