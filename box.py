from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# モデル
model = YOLO("/Users/chinenyoshinori/congestion-1/runs/detect/train16/weights/best.pt")  

# 画像
image_path = "/Users/chinenyoshinori/congestion-1/data/add.images/image_51.jpg"  

# 画像の読み込み
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCVのBGRからRGBに変換

results = model(image)

# クラスID 2（車）のみを抽出
car_detections = [det for det in results[0].boxes.data if int(det[5]) == 2]

# 描画
for det in car_detections:
    x1, y1, x2, y2, conf, cls = det  # バウンディングボックス情報
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # バウンディングボックス）
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    label = f"Car: {conf:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 保存
output_path = "car_detection_result.jpg"
cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.axis("off")
plt.show()

print(f"保存された画像: {output_path}")



