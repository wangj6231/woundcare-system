from ultralytics import YOLO

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    # 載入剛剛訓練好的 150 輪最佳分類權重 (v3 版本)
    model = YOLO("runs/classify/wound_classifier_v3/weights/best.pt")
    
    print("🧪 正在對獨立測試集 (Test Set) 進行最終盲測評估...")
    
    # 將資料夾指向 v3 裡面的 test 目錄
    # split='test' 告訴 YOLO 尋找 test 目錄內的資料
    metrics = model.val(data='yolo_wound_cls_dataset_v3', split='test')
    
    # 印出最終測試成績
    top1_acc = metrics.top1
    top5_acc = metrics.top5
    print("\n" + "="*50)
    print("🏆 【最終盲測成績 (Blind Test Results)】")
    print(f"✅ Top-1 Accuracy (首選準確率): {top1_acc * 100:.2f}%")
    print("="*50)
    print("※ 這是模型自始至終完全沒有看過的 10% 獨立資料，此數據非常具有論文說服力！")
