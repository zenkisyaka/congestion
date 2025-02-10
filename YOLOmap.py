import torch
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from torchvision.ops import nms

# YOLOモデルをロード
model = YOLO("/Users/chinenyoshinori/congestion-1/runs/detect/train16/weights/best.pt")

# 推論画像のパス
image_path = "/Users/chinenyoshinori/congestion-1/data/add.images/image_51.jpg"
results = model(image_path)

# 推論結果を取得
predictions = results[0].boxes  

# 確率とバウンディングボックス情報を取得
scores = predictions.conf.cpu().numpy()  # GPU対応
classes = predictions.cls.cpu().numpy()  
bboxes = predictions.xyxy.cpu().numpy()  

# NMSを適用する関数
def apply_nms(bboxes, scores, iou_threshold=0.5):
    # NMSを適用するためにtorch tensorに変換
    bboxes_tensor = torch.tensor(bboxes)
    scores_tensor = torch.tensor(scores)
    
    # NMSの適用
    keep = nms(bboxes_tensor, scores_tensor, iou_threshold)
    
    # NMSを適用した後に残るボックス
    bboxes_after_nms = bboxes_tensor[keep].cpu().numpy()
    scores_after_nms = scores_tensor[keep].cpu().numpy()
    
    return bboxes_after_nms, scores_after_nms

# 確率マップを描画する関数
def plot_probability_map(image_path, bboxes, scores, classes, target_class):
    # 画像を読み込み
    image = plt.imread(image_path)

    # 画像のサイズ取得
    height, width, _ = image.shape

    # 空の確率マップを作成
    prob_map = np.zeros((height, width))

    # ターゲットクラスに対するバウンディングボックスをNMSでフィルタリング
    bboxes_after_nms, scores_after_nms = apply_nms(bboxes, scores, iou_threshold=0.5)

    # ターゲットクラスの確率をマッピング
    for bbox, score, cls in zip(bboxes_after_nms, scores_after_nms, classes):
        if int(cls) == target_class:  # クラスIDを整数に変換して比較
            x_min, y_min, x_max, y_max = map(int, bbox)

            # 画像の範囲内に収める
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(width, x_max), min(height, y_max)

            # 確率をマップに加算
            prob_map[y_min:y_max, x_min:x_max] += score

    # 確率マップの最大値でスケーリング
    prob_map = np.clip(prob_map, 0, 1)  # 0から1の範囲にクリップ
    prob_map = prob_map / prob_map.max()  # 最大値でスケーリング

    # 可視化
    plt.figure(figsize=(10, 10))
    plt.imshow(image, alpha=0.8)  # 元画像
    plt.imshow(prob_map, cmap='jet', alpha=0.5)  # 確率マップを重ねる
    plt.colorbar(label="Probability")
    plt.axis("off")
    plt.title(f"Probability Map for Class {target_class}")
    plt.show()

# ターゲットクラスを指定（例: 0 = 人, 1 = 車 など）
target_class = 2  # クラスIDに基づいて設定
plot_probability_map(image_path, bboxes, scores, classes, target_class)
