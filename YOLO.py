import torch

# MPS デバイスを取得
device = torch.device("mps")

# ダミーデータ
x = torch.rand(3, 3).to(device)  # データを MPS デバイスに移動
print("Data on MPS device:", x)

# シンプルなモデル
model = torch.nn.Linear(3, 1).to(device)  # モデルを MPS デバイスに移動
output = model(x)  # MPS 上で計算
print("Output:", output)

from ultralytics import YOLO

# 事前学習済みモデルをロード
model = YOLO("yolov8m.pt")

# カスタムデータセットで再学習
model.train(
    data="/Users/chinenyoshinori/congestion-1/dataset/data.yaml",  # データセット設定ファイル
    epochs=50,                 # エポック数
    imgsz=320,                 # 画像サイズ
    batch=16,                  # バッチサイズ
    workers=4                 # データローダのワーカー数
)
