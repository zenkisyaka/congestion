from ultralytics import YOLO

# YOLOv8モデルのロード
model = YOLO("yolov8m.pt")  # 事前学習済みモデル

model.train(
    data="dataset2/data.yaml",
    epochs=50,
    batch=2,
    imgsz=1920,  # 最大画像サイズを指定
    multi_scale=True,  # 可変サイズで学習
    device="cpu"
)