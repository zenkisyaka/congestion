from ultralytics import YOLO

# 学習済みモデルをロード
model = YOLO("/Users/chinenyoshinori/congestion-1/runs/detect/train16/weights/best.pt")

# モデルの評価（検証セットを使用）
metrics = model.val()
