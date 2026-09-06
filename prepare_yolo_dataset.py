import os
import cv2
import shutil
import yaml

def convert_mask_to_yolo(mask_path, output_txt_path, class_id=0):
    """讀取單張遮罩並轉換為 YOLO 邊界框格式寫入 txt"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
        
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    with open(output_txt_path, 'w') as f:
        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            norm_w = w / width
            norm_h = h / height
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    return True

def process_folder(img_dir, mask_dir, dest_img_dir, dest_lbl_dir):
    """處理特定資料夾 (例如將 train_images 轉入 yolo/images/train)"""
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_lbl_dir, exist_ok=True)
    
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    count = 0
    
    for img_f in img_files:
        base_name = os.path.splitext(img_f)[0]
        # data_wound_seg 裡面的 mask 通常是 .png 或同名檔
        mask_f = base_name + ".png"
        mask_path = os.path.join(mask_dir, mask_f)
        if not os.path.exists(mask_path):
            mask_f = img_f
            mask_path = os.path.join(mask_dir, mask_f)
            
        if os.path.exists(mask_path):
            # 複製圖片
            shutil.copy(os.path.join(img_dir, img_f), os.path.join(dest_img_dir, img_f))
            # 轉換 Mask 到 YOLO Txt
            txt_name = base_name + ".txt"
            convert_mask_to_yolo(mask_path, os.path.join(dest_lbl_dir, txt_name))
            count += 1
            
    print(f"✅ 處理完成: 共轉換 {count} 張圖片與標籤")

def prepare_data_wound_seg(dataset_root, output_dir):
    print("🚀 開始處理 Kaggle Wound Segmentation 資料集...")
    
    # 輸入路徑 (Kaggle 解壓縮後的資料夾)
    train_images = os.path.join(dataset_root, "train_images")
    train_masks = os.path.join(dataset_root, "train_masks")
    test_images = os.path.join(dataset_root, "test_images")
    test_masks = os.path.join(dataset_root, "test_masks")
    
    # 輸出路徑 (YOLO 標準格式)
    yolo_train_img = os.path.join(output_dir, "images", "train")
    yolo_train_lbl = os.path.join(output_dir, "labels", "train")
    yolo_val_img = os.path.join(output_dir, "images", "val")
    yolo_val_lbl = os.path.join(output_dir, "labels", "val")
    
    print("📦 正在處理訓練集 (Train)...")
    process_folder(train_images, train_masks, yolo_train_img, yolo_train_lbl)
    
    print("📦 正在處理驗證集 (Val / Test)...")
    process_folder(test_images, test_masks, yolo_val_img, yolo_val_lbl)
    
    # 生成 dataset.yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    yaml_data = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "Wound"}
    }
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)
        
    print(f"🎉 全部轉換成功！您的 YOLO 資料集已經準備好在 {output_dir}")

if __name__ == "__main__":
    # 將您的 Kaggle Segmentation 資料夾指向正確路徑
    DATA_ROOT = "data_wound_seg"
    YOLO_OUTPUT = "yolo_wound_dataset"
    
    prepare_data_wound_seg(DATA_ROOT, YOLO_OUTPUT)
