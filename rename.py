import os
import shutil

def rename_and_copy_images(source_directory, target_directory, start_index=51):
    """
    指定したフォルダ内の画像をリネームし、別フォルダにコピーする。

    Args:
        source_directory (str): 画像が格納されているフォルダのパス（元のフォルダ）
        target_directory (str): リネーム後の画像を保存するフォルダ
        start_index (int): 変更後のファイル名の開始番号（デフォルトは51）
    """
    # 出力フォルダが存在しなければ作成
    os.makedirs(target_directory, exist_ok=True)

    # フォルダ内のファイルを取得
    files = sorted(os.listdir(source_directory))  # ソートして順番に変更
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}  # 画像の拡張子リスト
    index = start_index

    for file_name in files:
        source_path = os.path.join(source_directory, file_name)
        
        # ファイルかつ画像形式ならリネーム
        if os.path.isfile(source_path) and os.path.splitext(file_name)[1].lower() in image_extensions:
            new_name = f"image_{index}.jpg"  # `image_51.jpg`, `image_52.jpg` のようにリネーム
            target_path = os.path.join(target_directory, new_name)

            # 画像をコピー & リネーム
            shutil.copy2(source_path, target_path)
            print(f"✅ {file_name} → {new_name}（{target_directory} に保存）")

            index += 1

# 📂 フォルダのパスを指定（★ 変更して使う）
source_folder = "/Users/chinenyoshinori/congestion-1/data/add.images"  # 元のフォルダ
target_folder = "/Users/chinenyoshinori/congestion-1/data/add.images"  # 保存先フォルダ

rename_and_copy_images(source_folder, target_folder)
