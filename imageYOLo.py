from ultralytics import YOLO

# YOLOモデルのロード（事前学習済み or トレーニング済みのモデル）
model = YOLO("/Users/chinenyoshinori/congestion-1/runs/detect/train16/weights/best.pt")

# 推論する画像のパス
image_path = "/Users/chinenyoshinori/congestion-1/data/backup/detection/image_.jpg"

# 画像に対する推論実行
results = model(image_path)

# 検出されたバウンディングボックス、スコア、クラスを取得
predictions = results[0].boxes
scores = predictions.conf.cpu().numpy()  
classes = predictions.cls.cpu().numpy()
bboxes = predictions.xyxy.cpu().numpy()

# 最大の信頼度スコアとそのインデックスを取得
max_score = max(scores)
max_index = scores.index(max_score)

# 最大の信頼度を持つ物体のクラスとバウンディングボックス
max_class = classes[max_index]
max_bbox = bboxes[max_index]

# 結果を表示
print(f"最大の信頼度スコア: {max_score}")
print(f"最大の信頼度を持つ物体のクラス: {max_class}")
print(f"最大の信頼度を持つ物体のバウンディングボックス: {max_bbox}")
