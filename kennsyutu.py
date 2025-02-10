import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1️⃣ モデルのロード（YOLOv8 の学習済みモデル）
model_path = "/Users/chinenyoshinori/congestion-1/runs/detect/train16/weights/best.pt"
model = YOLO(model_path)

# 2️⃣ 画像の読み込み
image_path = "/Users/chinenyoshinori/congestion-1/data/add.images/image_51.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV の BGR 形式を RGB に変換
height, width, _ = image.shape

# 3️⃣ 保存フォルダを作成（なければ作る）
save_dir = "/Users/chinenyoshinori/congestion-1/yolo_results"
os.makedirs(save_dir, exist_ok=True)  # フォルダがなければ作成

# 4️⃣ YOLO で物体検出
results = model(image)

# 5️⃣ 検出結果をリスト化（テキスト保存用）
detections = []

# 6️⃣ 検出結果を画像に描画
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # バウンディングボックスの座標 (左上, 右下)
        conf = float(box.conf[0])  # 信頼度（Confidence Score）
        cls = int(box.cls[0])  # クラスID
        class_name = model.names[cls]  # クラス名（ラベル）

        # バウンディングボックスの描画（赤枠）
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 検出結果をリストに保存
        detections.append(f"{class_name} {conf:.2f} {x1} {y1} {x2} {y2}")

# 7️⃣ 検出結果を TXT ファイルに保存
txt_output_path = os.path.join(save_dir, "yolo_predictions.txt")
with open(txt_output_path, "w") as f:
    for detection in detections:
        f.write(detection + "\n")
print(f"📄 検出結果のテキストファイルを保存しました: {txt_output_path}")

# 8️⃣ 検出結果を画像として保存
output_image_path = os.path.join(save_dir, "yolo_prediction.jpg")
cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
print(f"📷 検出結果の画像を保存しました: {output_image_path}")

# 9️⃣ 画像を表示
plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.axis("off")
plt.show()


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ファイルパス
image_path = "/Users/chinenyoshinori/congestion-1/data/add.images/image_51.jpg"
prediction_path = "/Users/chinenyoshinori/congestion-1/yolo_results/yolo_predictions.txt"
annotation_path = "/Users/chinenyoshinori/congestion-1/ano_51-60/image_51.txt"

# 保存フォルダを作成
save_dir = "/Users/chinenyoshinori/congestion-1/yolo_comparison_results"
os.makedirs(save_dir, exist_ok=True)

# 画像を読み込む
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = image.shape

# YOLO フォーマット（正規化された座標）を (x1, y1, x2, y2) に変換
def yolo_to_bbox(yolo_bbox, img_width, img_height):
    x_center, y_center, w, h = yolo_bbox
    x1 = int((x_center - w / 2) * img_width)
    y1 = int((y_center - h / 2) * img_height)
    x2 = int((x_center + w / 2) * img_width)
    y2 = int((y_center + h / 2) * img_height)
    return x1, y1, x2, y2

# ファイルを読み込む関数
def load_bboxes(file_path):
    bboxes = []
    if not os.path.exists(file_path):
        print(f"⚠️ ファイルが見つかりません: {file_path}")
        return bboxes  # 空リストを返す

    with open(file_path, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            if len(values) == 6:  # YOLO の検出結果
                class_id, x, y, w, h, conf = values
            else:  # アノテーション
                class_id, x, y, w, h = values
                conf = 1.0  # アノテーションは信頼度 1.0

            bbox = yolo_to_bbox((x, y, w, h), width, height)
            bboxes.append((class_id, bbox, conf))
    return bboxes

# IoU（Intersection over Union）の計算
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou, (xi1, yi1, xi2, yi2)  # Intersection の座標も返す

# バウンディングボックスを読み込み
predictions = load_bboxes(prediction_path)
annotations = load_bboxes(annotation_path)

# 検出結果をリスト化（IoU 保存用）
iou_results = []

# 検出結果を描画
for _, pred_box, conf in predictions:
    x1, y1, x2, y2 = pred_box
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 青色
    cv2.putText(image, f"Pred {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# アノテーションを描画
for _, gt_box, _ in annotations:
    x1, y1, x2, y2 = gt_box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 緑色
    cv2.putText(image, "GT", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 各予測に対する IoU を計算して表示
for _, pred_box, _ in predictions:
    best_iou = 0
    best_intersection = None
    for _, gt_box, _ in annotations:
        iou, intersection = compute_iou(pred_box, gt_box)
        if iou > best_iou:
            best_iou = iou
            best_intersection = intersection

    x1, y1, _, _ = pred_box
    cv2.putText(image, f"IoU: {best_iou:.2f}", (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # IoU の重なりを半透明の塗りつぶしで表示
    if best_intersection is not None:
        xi1, yi1, xi2, yi2 = best_intersection
        overlay = image.copy()
        alpha = best_iou * 0.7  # IoU が高いほど濃くなる
        cv2.rectangle(overlay, (xi1, yi1), (xi2, yi2), (255, 255, 0), -1)  # 黄色
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.putText(image, "Overlap", (xi1, yi1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

# IoU の結果を TXT に保存
iou_txt_path = os.path.join(save_dir, "iou_results.txt")
with open(iou_txt_path, "w") as f:
    for line in iou_results:
        f.write(line + "\n")
print(f"📄 IoU の結果を保存しました: {iou_txt_path}")

# 結果を保存
output_image_path = os.path.join(save_dir, "bbox_comparison_with_overlap.jpg")
cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
print(f"📷 検出結果の画像を保存しました: {output_image_path}")

# 結果を表示
plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.axis("off")
plt.show()
