import os

def rename_images(folder_path, prefix="image"):
    """
    指定したフォルダ内の画像ファイルの名前を単純な名前にリネームする。
    :param folder_path: 画像が保存されているフォルダのパス
    :param prefix: ファイル名の接頭辞（デフォルトは "image"）
    """
    # フォルダ内のすべてのファイルを取得
    files = os.listdir(folder_path)
    
    # 画像ファイルだけをフィルタリング
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # リネーム処理
    for index, old_name in enumerate(image_files):
        # 拡張子を取得
        extension = os.path.splitext(old_name)[1]
        
        # 新しい名前を生成
        new_name = f"{prefix}_{index + 1}{extension}"
        
        # 古いパスと新しいパスを作成
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)
        
        # ファイルをリネーム
        os.rename(old_path, new_path)
        print(f"Renamed: {old_name} -> {new_name}")

# 実行例
image_folder = "/Users/chinenyoshinori/congestion-1/data/images"  # 画像フォルダのパス
rename_images(image_folder)

import os
import cv2
from ultralytics import YOLO

# YOLOモデルをロード
model = YOLO("yolov8m.pt")

# クラス名リスト
class_names = model.names  # YOLOモデルからクラス名を取得

# YOLOの処理を関数化
def yolo_judge(model, image_path):
    results = model(image_path)
    return results

def save_labels_and_images(results, image_file, image_path, output_folder, bbox_folder, target_class_id=2):
    """
    推論結果をフィルタリングし、クラスID 2（car）のみを処理。
    """
    # ラベルファイルの保存
    label_file = os.path.join(output_folder, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
    with open(label_file, "w") as f:
        for r in results:
            for box in r.boxes.data:  # YOLOv8では .data を使用
                x1, y1, x2, y2, conf, cls = box[:6]
                if int(cls) == target_class_id:  # クラスIDが2（car）のみ処理
                    image_width, image_height = 800, 600  # 必要に応じて画像サイズを設定
                    x_center = ((x1 + x2) / 2) / image_width
                    y_center = ((y1 + y2) / 2) / image_height
                    width = (x2 - x1) / image_width
                    height = (y2 - y1) / image_height
                    f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # バウンディングボックスを描画した画像を保存
    img = cv2.imread(image_path)
    for r in results:
        for box in r.boxes.data:
            x1, y1, x2, y2, conf, cls = map(int, box[:6])  # バウンディングボックス座標を整数化
            if int(cls) == target_class_id:  # クラスIDが2（car）のみ描画
                label = f"{int(cls)}: {conf:.2f}"  # クラスIDのみ表示
                color = (0, 255, 0)  # 緑色のボックス
                # バウンディングボックスとラベルの描画
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 処理後の画像を保存
    bbox_image_path = os.path.join(bbox_folder, image_file)
    cv2.imwrite(bbox_image_path, img)

# フォルダ内の全ての画像を処理
image_folder = "/Users/chinenyoshinori/congestion-1/data/images"  # 画像フォルダのパス
output_folder = "/Users/chinenyoshinori/congestion-1/data/labels"  # ラベルファイル保存先フォルダ
bbox_folder = "/Users/chinenyoshinori/congestion-1/data/YOLO_images"  # バウンディングボックス画像保存先フォルダ

os.makedirs(output_folder, exist_ok=True)  # ラベルフォルダが存在しない場合は作成
os.makedirs(bbox_folder, exist_ok=True)  # バウンディングボックス画像フォルダが存在しない場合は作成

image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]  # フォルダ内の画像リスト

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    print(f"Processing: {image_path}")
    results = yolo_judge(model, image_path)
    save_labels_and_images(results, image_file, image_path, output_folder, bbox_folder, target_class_id=2)
    print(f"Labels and bounding box image saved for {image_file}")



import os
import random
import shutil

def save_class_file(class_names, output_folder):
    """
    クラス名をclasses.txtに保存
    """
    os.makedirs(output_folder, exist_ok=True)  # フォルダがない場合は作成
    class_file_path = os.path.join(output_folder, "classes.txt")
    with open(class_file_path, "w") as f:
        for class_id, class_name in class_names.items():
            f.write(f"{class_name}\n")
    print(f"Saved classes.txt in {output_folder}")

def split_files_with_bounding_boxes(
    input_image_folder, input_label_folder, input_bbox_image_folder, 
    output_train_folder, output_test_folder, class_names, train_ratio=0.8):
    """
    YOLOで検出したバウンディングボックスの画像も含めて複製で分割する。
    
    :param input_image_folder: 元の画像フォルダ
    :param input_label_folder: ラベルフォルダ
    :param input_bbox_image_folder: バウンディングボックス描画済み画像フォルダ
    :param output_train_folder: トレインデータ保存フォルダ
    :param output_test_folder: テストデータ保存フォルダ
    :param class_names: YOLOモデルのクラス名
    :param train_ratio: トレインデータの割合（デフォルト: 0.8）
    """
    # 入力フォルダ内の画像ファイルを取得
    image_files = [f for f in os.listdir(input_image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # ファイルをシャッフル
    random.shuffle(image_files)
    
    # 分割位置を計算
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    test_files = image_files[split_index:]
    
    # トレイン用とテスト用のフォルダを作成
    os.makedirs(os.path.join(output_train_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_train_folder, "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_train_folder, "bbox_images"), exist_ok=True)
    os.makedirs(os.path.join(output_test_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_test_folder, "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_test_folder, "bbox_images"), exist_ok=True)

    # トレイン用フォルダにデータを複製
    for file in train_files:
        # 元画像
        image_path = os.path.join(input_image_folder, file)
        # ラベルデータ
        label_path = os.path.join(input_label_folder, file.replace('.jpg', '.txt').replace('.png', '.txt'))
        # バウンディングボックス描画済み画像
        bbox_image_path = os.path.join(input_bbox_image_folder, file)

        # 複製
        shutil.copy(image_path, os.path.join(output_train_folder, "images", file))
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(output_train_folder, "labels", os.path.basename(label_path)))
        if os.path.exists(bbox_image_path):
            shutil.copy(bbox_image_path, os.path.join(output_train_folder, "bbox_images", file))
    
    # テスト用フォルダにデータを複製
    for file in test_files:
        # 元画像
        image_path = os.path.join(input_image_folder, file)
        # ラベルデータ
        label_path = os.path.join(input_label_folder, file.replace('.jpg', '.txt').replace('.png', '.txt'))
        # バウンディングボックス描画済み画像
        bbox_image_path = os.path.join(input_bbox_image_folder, file)

        # 複製
        shutil.copy(image_path, os.path.join(output_test_folder, "images", file))
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(output_test_folder, "labels", os.path.basename(label_path)))
        if os.path.exists(bbox_image_path):
            shutil.copy(bbox_image_path, os.path.join(output_test_folder, "bbox_images", file))
    
    # クラス名を保存
    save_class_file(class_names, output_train_folder)
    save_class_file(class_names, output_test_folder)

    print(f"データをランダムに分割しました！（複製）")
    print(f"トレインデータ: {len(train_files)} ファイル")
    print(f"テストデータ: {len(test_files)} ファイル")

# 実行例
input_image_folder = "/Users/chinenyoshinori/congestion-1/data/images"  # 元の画像フォルダ
input_label_folder = "/Users/chinenyoshinori/congestion-1/data/labels"  # ラベルフォルダ
input_bbox_image_folder = "/Users/chinenyoshinori/congestion-1/data/YOLO_images"  # バウンディングボックス描画済み画像フォルダ
output_train_folder = "/Users/chinenyoshinori/congestion-1/dataset/train"  # トレインデータ保存フォルダ
output_test_folder = "/Users/chinenyoshinori/congestion-1/dataset/val"   # テストデータ保存フォルダ

# クラス名をYOLOモデルから取得
from ultralytics import YOLO
model = YOLO("yolov8m.pt")
class_names = model.names

split_files_with_bounding_boxes(
    input_image_folder, input_label_folder, input_bbox_image_folder, 
    output_train_folder, output_test_folder, class_names)



import os

def normalize_labels(label_folder, image_width, image_height):
    for label_file in os.listdir(label_folder):
        if label_file.endswith(".txt"):
            file_path = os.path.join(label_folder, label_file)
            with open(file_path, "r") as f:
                lines = f.readlines()

            normalized_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x_center, y_center, width, height = map(float, parts)
                    # 正規化を適用
                    x_center /= image_width
                    y_center /= image_height
                    width /= image_width
                    height /= image_height

                    # 値が0～1の範囲に収まるように調整
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))

                    normalized_lines.append(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            # 上書き保存
            with open(file_path, "w") as f:
                f.writelines(normalized_lines)

            print(f"修正済み: {file_path}")

# 使用例
label_folder = "/path/to/val/labels"  # ラベルフォルダ
image_width = 1280  # 画像の幅
image_height = 720  # 画像の高さ
normalize_labels(label_folder, image_width, image_height)

import os
import shutil

def merge_annotation_and_detection_keep_original(annotation_folder, detection_folder, output_folder):
    # フォルダを作成（出力フォルダがない場合）
    os.makedirs(output_folder, exist_ok=True)

    # アノテーションファイルと検出結果ファイルを処理
    for annotation_file in os.listdir(annotation_folder):
        if annotation_file.endswith(".txt"):  # テキストファイルのみ処理
            annotation_path = os.path.join(annotation_folder, annotation_file)
            detection_path = os.path.join(detection_folder, annotation_file)  # 同じ名前のファイルを探す
            output_path = os.path.join(output_folder, annotation_file)

            # アノテーションファイルの内容を読み込み
            with open(annotation_path, "r", encoding="utf-8") as f:
                annotation_lines = f.readlines()

            # 検出結果ファイルの内容を読み込み（存在する場合のみ）
            detection_lines = []
            if os.path.exists(detection_path):
                with open(detection_path, "r", encoding="utf-8") as f:
                    detection_lines = f.readlines()

            # アノテーションと検出結果を結合
            combined_lines = annotation_lines + detection_lines

            # 結果を出力ファイルに保存
            with open(output_path, "w", encoding="utf-8") as f:
                f.writelines(combined_lines)

            print(f"結合済み: {output_path}")

    # 元のデータをバックアップフォルダに保存（任意で有効化）
    backup_folder = os.path.join(output_folder, "backup")
    os.makedirs(backup_folder, exist_ok=True)
    shutil.copytree(annotation_folder, os.path.join(backup_folder, "annotation"), dirs_exist_ok=True)
    shutil.copytree(detection_folder, os.path.join(backup_folder, "detection"), dirs_exist_ok=True)
    print(f"元のデータをバックアップフォルダに保存しました: {backup_folder}")

# フォルダパスと出力ファイルパスを指定
annotation_folder = "/Users/chinenyoshinori/congestion-1/dataset/train/ano_labels"  # アノテーションファイルのフォルダ
detection_folder = "dataset/train/labels"    # YOLO検出結果のフォルダ
output_folder = "/Users/chinenyoshinori/congestion-1/dataset2"          # 結合結果の保存先フォルダ

# 関数を実行
merge_annotation_and_detection_keep_original(annotation_folder, detection_folder, output_folder)


from ultralytics import YOLO

# 1. モデルのロード（事前学習済みモデルを指定）
model = YOLO("yolov8m.pt")  # YOLOv8の事前学習済みモデル

# 2. トレーニング
print("---- トレーニング開始 ----")
train_results = model.train(
    data="/Users/chinenyoshinori/congestion-1/dataset2/data.yaml",  # データセットのYAMLファイル
    epochs=50,                     # エポック数
    imgsz=640,                     # 画像サイズ
    batch=16,                      # バッチサイズ（調整可能）
    device='cpu'        # 使用するデバイス（0はGPU、'cpu'ならCPU）
)
print("---- トレーニング完了 ----")

# 3. 検証（自動で検証データセットを使用）
print("---- 検証開始 ----")
val_results = model.val(
    data="/Users/chinenyoshinori/congestion-1/dataset2/data.yaml",  # データセットのYAMLファイル
    imgsz=640                      # 画像サイズ
)
print("---- 検証完了 ----")


import os
from ultralytics import YOLO

import os

def fix_invalid_labels(label_folder, max_class_id):
    for label_file in os.listdir(label_folder):
        if label_file.endswith(".txt"):
            file_path = os.path.join(label_folder, label_file)
            with open(file_path, "r") as f:
                lines = f.readlines()

            valid_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x_center, y_center, width, height = map(float, parts)
                    if 0 <= cls <= max_class_id:  # 有効なクラスIDか確認
                        valid_lines.append(line)

            # 修正されたラベルを書き込み
            with open(file_path, "w") as f:
                f.writelines(valid_lines)

            print(f"修正済み: {file_path}")

# 使用例
train_labels = "/Users/chinenyoshinori/congestion-1/dataset2/train/labels"
val_labels = "/Users/chinenyoshinori/congestion-1/dataset2/val/labels"
max_class_id = 2  # データセットの最大クラスID
fix_invalid_labels(train_labels, max_class_id)
fix_invalid_labels(val_labels, max_class_id)


# 1. ラベルの正規化関数
def normalize_labels(label_folder, image_width, image_height):
    for label_file in os.listdir(label_folder):
        if label_file.endswith(".txt"):
            file_path = os.path.join(label_folder, label_file)
            with open(file_path, "r") as f:
                lines = f.readlines()

            normalized_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x_center, y_center, width, height = map(float, parts)
                    # 正規化
                    x_center /= image_width
                    y_center /= image_height
                    width /= image_width
                    height /= image_height

                    # 値が0～1の範囲に収まるように調整
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))

                    normalized_lines.append(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            # 上書き保存
            with open(file_path, "w") as f:
                f.writelines(normalized_lines)

            print(f"修正済み: {file_path}")

# 2. 空のラベルファイルを作成する関数
def create_empty_labels(image_folder, label_folder):
    os.makedirs(label_folder, exist_ok=True)
    for image_file in os.listdir(image_folder):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            label_file = os.path.join(label_folder, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
            if not os.path.exists(label_file):
                with open(label_file, 'w') as f:
                    pass  # 空のラベルファイルを作成
                print(f"空のラベルファイルを作成: {label_file}")

# 3. ラベル処理を適用
def process_labels(train_folder, val_folder, image_width, image_height):
    print("---- ラベルの正規化開始 ----")
    normalize_labels(os.path.join(train_folder, "labels"), image_width, image_height)
    normalize_labels(os.path.join(val_folder, "labels"), image_width, image_height)

    print("---- 空のラベル作成開始 ----")
    create_empty_labels(os.path.join(train_folder, "images"), os.path.join(train_folder, "labels"))
    create_empty_labels(os.path.join(val_folder, "images"), os.path.join(val_folder, "labels"))

# 4. トレーニングと検証
def train_and_validate(model_path, data_yaml, epochs, imgsz, batch, device):
    model = YOLO(model_path)

    print("---- トレーニング開始 ----")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device
    )
    print("---- トレーニング完了 ----")

    print("---- 検証開始 ----")
    model.val(
        data=data_yaml,
        imgsz=imgsz
    )
    print("---- 検証完了 ----")

# メイン処理
if __name__ == "__main__":
    # パスの設定
    train_folder = "/Users/chinenyoshinori/congestion-1/dataset2/train"
    val_folder = "/Users/chinenyoshinori/congestion-1/dataset2/val"
    data_yaml = "/Users/chinenyoshinori/congestion-1/dataset2/data.yaml"
    model_path = "yolov8m.pt"

    # パラメータの設定
    image_width = 1280
    image_height = 720
    epochs = 50
    imgsz = 640
    batch = 16
    device = 'cpu'  # GPUが利用できない場合は 'cpu'

    # ラベル処理
    process_labels(train_folder, val_folder, image_width, image_height)

    # トレーニングと検証
    train_and_validate(model_path, data_yaml, epochs, imgsz, batch, device)


