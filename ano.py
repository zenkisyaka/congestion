from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# モデルのロード（YOLOv8の事前学習済みモデルを使用）
model = YOLO("/Users/chinenyoshinori/congestion-1/runs/detect/train16/weights/best.pt")  # トレーニング済みモデル

# 画像のパス（適宜変更）
image_path = "/Users/chinenyoshinori/congestion-1/data/add.images/image_51.jpg"  # 画像ファイルを指定

# 画像の読み込み
image = cv2.imread(image_path)
original_image = image.copy()  # 元の画像を保存しておく（サイズ変換の影響を受けないように）
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCVのBGRからRGBに変換

# YOLOで物体検出を実行
results = model(image)

# 車（クラスID 2）のみを抽出
car_detections = [det for det in results[0].boxes if int(det.cls) == 2]

if not car_detections:
    print("車が検出されませんでした。")
else:
    # 信頼度が最も高い検出を選ぶ
    max_confidence_detection = max(car_detections, key=lambda det: det.conf)  # 最大の信頼度を持つ検出結果

    # 最大の信頼度を持つ検出結果のバウンディングボックスを取得
    xyxy = max_confidence_detection.xyxy.cpu().numpy()  # バウンディングボックス座標を取得し、テンソルからNumPy配列に変換

    # `xyxy` 配列が1行（1つの検出）の場合にアクセスする
    x1, y1, x2, y2 = map(int, xyxy[0])  # `xyxy[0]` で最初の検出結果を選択

    conf = max_confidence_detection.conf  # 信頼度
    cls = max_confidence_detection.cls  # クラスID

    # 結果の表示
    print(f"最大の信頼度スコア: {conf:.2f}")
    print(f"バウンディングボックス座標: [{x1}, {y1}, {x2}, {y2}]")

    # バウンディングボックスを描画（赤枠）
    cv2.rectangle(original_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
    label = f"Car: {conf:.2f}"
    cv2.putText(original_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 結果を保存
    output_path = "car_detection_result.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))

    # 結果を表示
    plt.figure(figsize=(10, 6))
    plt.imshow(original_image)
    plt.axis("off")
    plt.show()

    # 保存された画像のパスを出力
    print(f"保存された画像: {output_path}")
