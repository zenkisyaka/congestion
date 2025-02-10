import cv2
from ultralytics import YOLO

# 動画
video_path = '/Users/chinenyoshinori/congestion-1/data/movie/760508870.377439.mp4'

# モデル
model = YOLO('/Users/chinenyoshinori/congestion-1/runs/detect/train16/weights/best.pt')

# 動画ファイルの読み込み
cap = cv2.VideoCapture(video_path)

output_path = '/Users/chinenyoshinori/congestion-1/data/movie'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # フレームごとに物体検知を行う
        results = model(frame)
        
        # 描画
        annotated_frame = results[0].plot()
        
        # 出力動画にフレームを書き込む
        out.write(annotated_frame)
        
        # フレームを表示
        cv2.imshow('Frame', annotated_frame)
        
        # 'z'が押されたら終了
        if cv2.waitKey(1) & 0xFF == ord('z'):
            break
    else:
        break

# リソースの解放
cap.release()
out.release()
cv2.destroyAllWindows()
