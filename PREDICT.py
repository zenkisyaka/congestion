from ultralytics import YOLO

# 再学習させたモデル
model = YOLO("/Users/chinenyoshinori/congestion-1/runs/detect/train6/weights/best.pt")

# 検出実行
results = model.predict(source="/Users/chinenyoshinori/congestion-1/data/noimages/LINE_ALBUM_YoLo用画像_250116_1.jpg", save=True, conf=0.25)

# 推論結果を表示
for result in results:
    print(result.boxes)

    
