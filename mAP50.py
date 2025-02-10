import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from ultralytics import YOLO

# YOLOモデルをロード
model = YOLO("/Users/chinenyoshinori/congestion-1/runs/detect/train16/weights/best.pt")

# 推論画像のパス
image_path = "/Users/chinenyoshinori/congestion-1/data/add.images/image_51.jpg"
results = model(image_path)

# 推論結果を取得
predictions = results[0].boxes  # バウンディングボックス

# クラスID, スコア（信頼度）, バウンディングボックス座標を取得
scores = predictions.conf.cpu().numpy()  # スコア
classes = predictions.cls.cpu().numpy()  # クラスID
bboxes = predictions.xyxy.cpu().numpy()  # バウンディングボックス座標

# 実際のクラスラベル（Ground Truth）
# これは実際の検出結果に基づいて手動で準備する必要があります
# ここでは仮のデータを使用していますが、実際のGTラベルを使用してください
y_true = np.random.randint(0, 3, len(classes))  # 仮のクラスラベル（実際にはGTデータが必要）

# Precision-Recall曲線を計算し、AP（平均精度）を求める関数
def calculate_ap(classes, scores, y_true, num_classes):
    aps = []
    for cls in range(num_classes):
        # クラスごとのPrecision-Recall曲線を計算
        y_true_cls = (y_true == cls).astype(int)  # クラスごとに1, 他は0
        precision, recall, _ = precision_recall_curve(y_true_cls, scores)
        ap = auc(recall, precision)  # APはAUCとして計算
        aps.append(ap)
    return np.mean(aps)  # mAP（全クラスの平均AP）

# クラス数を設定（例: 0 = 人, 1 = 車, 2 = バイク）
num_classes = 3

# mAPを計算
mAP = calculate_ap(classes, scores, y_true, num_classes)
print(f'mAP: {mAP:.2f}')

# Precision-Recall曲線のプロット
plt.figure(figsize=(8, 6))
for cls in range(num_classes):
    # 各クラスのPrecision-Recall曲線を描画
    y_true_cls = (y_true == cls).astype(int)
    precision, recall, _ = precision_recall_curve(y_true_cls, scores)
    plt.plot(recall, precision, label=f'Class {cls}')
    
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (mAP = {mAP:.2f})')
plt.legend(loc='best')
plt.grid(True)
plt.show()
