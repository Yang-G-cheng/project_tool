from ultralytics import YOLO
import cv2
use_webcam = False
model = YOLO("yolo11n-seg.pt")
# source = 
# model.predict(source, save = True)
# 開啟 webcam
if use_webcam:
    cap = cv2.VideoCapture(0)  # 0 表示預設 webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # 即時處理迴圈
    while True:
        ret, frame = cap.read()  # 讀取一幀
        if not ret:
            print("Error: Failed to capture image.")
            break

        # 使用 YOLOv11 進行推論
        results = model.predict(source=frame, conf=0.25, imgsz=640)

        # 將結果繪製到影像上
        annotated_frame = results[0].plot()  # 繪製邊界框、類別等

        # 顯示結果
        cv2.imshow("YOLOv11 Webcam", annotated_frame)

        # 按 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()
else:
    source = 'walk.mp4'
    model.predict(source, save = True, project='custom_output' ,name='webcam_results')