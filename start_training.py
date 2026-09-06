from ultralytics import YOLO

# 1. 載入模型 (這裡使用 YOLOv8n 的預訓練模型開始)
model = YOLO('yolov8n.pt')

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    # 2. 開始訓練
    print("🚀 開始 YOLO 訓練任務...")
    results = model.train(
        data='yolo_wound_dataset/dataset.yaml',  # 指向剛剛整理好的 YAML
        epochs=50,                               # 訓練輪數
        imgsz=640,                               # 影像大小
        batch=16,                                # 批次大小 (若顯示卡記憶體不足，請改為 8 或 4)
        name='wound_model_v1'                    # 訓練結果的資料夾名稱
    )

    print("✅ 訓練完成！您可以在 runs/detect/wound_model_v1/weights/ 找到您的 best.pt 權重檔。")
