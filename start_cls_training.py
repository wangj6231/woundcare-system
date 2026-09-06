from ultralytics import YOLO

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    # 1. 載入分類專用模型 (注意字尾是 -cls.pt)
    model = YOLO('yolov8n-cls.pt')

    # 2. 開始訓練分類任務
    print("🚀 開始 YOLO 傷口分類訓練任務 (加入放大縮小與權重微調)...")
    results = model.train(
        data='yolo_wound_cls_dataset_v3',  # 使用 v3 資料集 (已解決資料不平衡)
        epochs=150,                      # 訓練輪數
        imgsz=224,
        batch=16,
        name='wound_classifier_v3',      # 存為 v3
        
        # --- 資料增強 (Data Augmentation) ---
        scale=0.5,                       # 【照片放大縮小】隨機縮放 ±50%
        fliplr=0.5,                      # 50% 機率水平翻轉
        
        # --- 權重與超參數調整 (Weights Adjustment) ---
        optimizer='auto',
        lr0=0.01,                        # 學習率
        weight_decay=0.0005              # 權重衰減，防止過擬合 (符合您的需求 1)
    )

    print("✅ 分類訓練完成！您的權重檔會存放在 runs/classify/wound_classifier_v3/weights/best.pt")
