import os
import shutil
import random

def prepare_classification_dataset(src_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    將分類影像資料集隨機切割成 train, val 與 test。
    並針對 train 資料集進行「類別平衡權重調整 (Oversampling)」，
    確保所有傷口類別在訓練時的圖片數量一致，解決資料不平衡問題。
    """
    print(f"🚀 開始整理並平衡分類資料集: {src_dir}")
    
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    
    # 第一次掃描：決定最大訓練圖片數，以最大數量為基準進行 Oversampling
    train_counts = []
    class_splits = {}
    
    for cls in classes:
        cls_src = os.path.join(src_dir, cls)
        images = [f for f in os.listdir(cls_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            continue
            
        random.seed(42)
        random.shuffle(images)
        
        train_end = int(len(images) * train_ratio)
        val_end = train_end + int(len(images) * val_ratio)
        
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]
        
        class_splits[cls] = (train_imgs, val_imgs, test_imgs)
        train_counts.append(len(train_imgs))

    if not train_counts:
        return
        
    max_train_count = max(train_counts)
    print(f"ℹ️ 最大訓練類別數量為 {max_train_count} 張，將對其他較少類別進行自動擴充 (Oversampling) 以平衡權重...")
    
    total_imgs = 0
    for cls, (train_imgs, val_imgs, test_imgs) in class_splits.items():
        cls_src = os.path.join(src_dir, cls)
        cls_train = os.path.join(train_dir, cls)
        cls_val = os.path.join(val_dir, cls)
        cls_test = os.path.join(test_dir, cls)
        
        os.makedirs(cls_train, exist_ok=True)
        os.makedirs(cls_val, exist_ok=True)
        os.makedirs(cls_test, exist_ok=True)
        
        # 複製 Val 和 Test (不干涉驗證與測試集)
        for img in val_imgs:
            shutil.copy(os.path.join(cls_src, img), os.path.join(cls_val, img))
        for img in test_imgs:
            shutil.copy(os.path.join(cls_src, img), os.path.join(cls_test, img))
            
        # 處理 Train 集的 Oversampling
        current_train_count = 0
        while current_train_count < max_train_count:
            for img in train_imgs:
                if current_train_count >= max_train_count:
                    break
                # 若重複抽取，檔名加上後綴避免覆蓋
                new_img_name = f"aug_{current_train_count}_{img}" if current_train_count >= len(train_imgs) else img
                shutil.copy(os.path.join(cls_src, img), os.path.join(cls_train, new_img_name))
                current_train_count += 1
                
        total_imgs += (current_train_count + len(val_imgs) + len(test_imgs))
        print(f"  📁 [{cls}]: 訓練={current_train_count} 張 (經擴充), 驗證={len(val_imgs)} 張, 測試={len(test_imgs)} 張")
        
    print(f"✅ 整理完成！所有訓練類別已達成完美平衡，總共處理了 {total_imgs} 張圖片，存於: {output_dir}")

if __name__ == '__main__':
    # 輸出到 v3 目錄
    prepare_classification_dataset("Wound_dataset", "yolo_wound_cls_dataset_v3")
