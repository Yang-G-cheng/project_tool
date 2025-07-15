from ultralytics import YOLO
import torch
import os
from pathlib import Path

def setup_training_environment():
    """設置訓練環境"""
    # 確保使用最佳設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 設置CUDA優化（如果使用GPU）
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    return device

def create_optimized_wafer_model():
    """創建針對wafer檢測優化的模型"""
    
    # 對於圓形邊緣檢測，建議使用segmentation模型
    # 但如果您想要detection，可以改為 "yolo11n.pt" 或 "yolo11s.pt"
    model = YOLO("yolo11n-seg.pt", task="segment")  # 使用預訓練權重
    
    return model

def train_wafer_model():
    """訓練wafer邊緣檢測模型"""
    
    # 設置環境
    device = setup_training_environment()
    
    # 創建模型
    model = create_optimized_wafer_model()
    
    # 針對wafer邊緣檢測的優化訓練參數
    training_args = {
        # 基本設定
        'data': 'my_seg.yaml',
        'epochs': 500,  # 減少epochs，避免過擬合
        'imgsz': 640,   # 640通常比1024更好，訓練更快且效果不差
        'batch': 8,     # 增加batch size以提高訓練穩定性
        'device': device,
        'workers': 4,   # 減少workers避免數據載入瓶頸
        
        # 預訓練設定
        'pretrained': True,  # 使用預訓練權重！重要改進
        
        # 優化器設定
        'optimizer': 'AdamW',
        'lr0': 0.002,        # 降低初始學習率，更穩定
        'lrf': 0.01,         # 最終學習率
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # 損失函數權重 - 針對邊緣檢測優化
        'box': 7.5,          # 邊界框損失權重
        'cls': 0.5,          # 分類損失權重  
        'dfl': 1.5,          # 分佈焦點損失權重
        
        # 數據增強 - 針對圓形wafer優化
        'hsv_h': 0.015,      # 色調變化
        'hsv_s': 0.7,        # 飽和度變化
        'hsv_v': 0.4,        # 明度變化
        'degrees': 0.0,      # 旋轉角度（圓形可以任意旋轉）
        'translate': 0.1,    # 平移
        'scale': 0.5,        # 縮放
        'shear': 0.0,        # 剪切（圓形不適用）
        'perspective': 0.0,  # 透視變換（圓形不適用）
        'flipud': 0.5,       # 上下翻轉
        'fliplr': 0.5,       # 左右翻轉
        'mosaic': 1.0,       # 馬賽克增強
        'mixup': 0.0,        # 混合增強
        'copy_paste': 0.0,   # 複製貼上增強
        
        # 訓練策略
        'patience': 100,     # 早停耐心值
        'save_period': 50,   # 保存週期
        'val': True,         # 啟用驗證
        'plots': True,       # 生成訓練圖表
        'save': True,        # 保存模型
        'save_txt': True,    # 保存預測結果
        'save_conf': True,   # 保存置信度
        
        # 驗證設定
        'val_period': 10,    # 驗證頻率
        'iou': 0.7,          # IoU閾值
        'conf': 0.001,       # 置信度閾值
        'max_det': 300,      # 最大檢測數量
        
        # 多尺度訓練
        'multi_scale': True,
        
        # 標籤平滑
        'label_smoothing': 0.0,
        
        # 學習率調度
        'cos_lr': True,      # 使用餘弦退火
        'warmup_epochs': 3,  # 熱身期
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # 模型結構
        'dropout': 0.0,      # Dropout率
        'freeze': 10,        # 凍結前10層
        
        # 輸出設定
        'project': 'runs/train',
        'name': 'wafer_edge_detection_optimized',
        'verbose': True,
        'exist_ok': True,
        
        # 記錄設定
        'amp': True,         # 自動混合精度
        'fraction': 1.0,     # 使用數據集的比例
        'profile': False,    # 性能分析
        
        # 分割特定參數（如果使用segmentation）
        'overlap_mask': True,
        'mask_ratio': 4,
        
        # 針對圓形檢測的anchor設定
        'anchor_t': 4.0,
        
        # 類別權重
        'cls_pw': 1.0,
        'obj_pw': 1.0,
        
        # 焦點損失
        'fl_gamma': 0.0,
        
        # 關閉一些不必要的功能以提高速度
        'save_hybrid': False,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'visualize': False,
        'augment': False,
        'agnostic_nms': False,
        'retina_masks': False,
        'format': 'torchscript',
        'keras': False,
        'optimize': False,
        'int8': False,
        'dynamic': False,
        'simplify': False,
        'opset': 17,
        'workspace': 4,
        'nms': False,
        'rect': False
    }
    
    print("開始訓練wafer邊緣檢測模型...")
    print(f"訓練參數: {training_args}")
    
    # 開始訓練
    results = model.train(**training_args)
    
    return results, model

def validate_model(model, data_yaml='my_seg.yaml'):
    """驗證模型性能"""
    print("\n開始模型驗證...")
    
    # 驗證參數
    val_args = {
        'data': data_yaml,
        'imgsz': 640,
        'batch': 1,
        'conf': 0.25,
        'iou': 0.7,
        'max_det': 300,
        'half': False,
        'device': '',
        'dnn': False,
        'plots': True,
        'rect': False,
        'save_txt': True,
        'save_conf': True,
        'save_json': False,
        'project': 'runs/val',
        'name': 'wafer_validation',
        'exist_ok': True,
        'split': 'val',
        'verbose': True
    }
    
    # 執行驗證
    metrics = model.val(**val_args)
    
    return metrics

def export_model(model, format='onnx'):
    """導出模型用於部署"""
    print(f"\n導出模型為 {format} 格式...")
    
    export_args = {
        'format': format,
        'imgsz': 640,
        'keras': False,
        'optimize': False,
        'half': False,
        'int8': False,
        'dynamic': False,
        'simplify': False,
        'opset': 17,
        'workspace': 4,
        'nms': False
    }
    
    # 導出模型
    model.export(**export_args)
    print(f"模型已導出為 {format} 格式")

def main():
    """主函數"""
    try:
        # 訓練模型
        results, model = train_wafer_model()
        
        print("\n訓練完成！")
        print(f"最佳模型保存在: {results.save_dir}")
        
        # 驗證模型
        metrics = validate_model(model)
        print(f"\n驗證指標:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        
        # 導出模型（可選）
        export_model(model, 'onnx')
        
        # 保存最終模型
        model.save('wafer_edge_detection_final.pt')
        print("最終模型已保存為 wafer_edge_detection_final.pt")
        
    except Exception as e:
        print(f"訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
