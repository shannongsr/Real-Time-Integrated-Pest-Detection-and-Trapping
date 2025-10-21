# eval_test.py
import argparse
import json
from pathlib import Path
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str,
                    default='runs/detect/yolov13_coffee_CIOU/weights/best.pt',
                    help='model weights path (best.pt or last.pt)')
    ap.add_argument('--data', type=str, default='mydata_caffee.yaml',
                    help='dataset yaml')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', type=str, default='0')
    ap.add_argument('--conf', type=float, default=0.001,
                    help='confidence threshold used for val (keep low for proper PR curves)')
    ap.add_argument('--iou', type=float, default=0.7,
                    help='NMS IoU threshold for val')
    ap.add_argument('--save_dir', type=str, default='runs/val_test',
                    help='directory to save reports')
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    # 关键：使用 split='test' 显式在 test 集评估
    metrics = model.val(
        data=args.data,
        split='val',
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        plots=True,         # 生成PR曲线、混淆矩阵等
        save_json=True,     # 导出COCO格式结果
        save_hybrid=False,
        save_dir=str(save_dir)
    )

    # ---- 总体指标 ----
    mp = metrics.box.mp  # mean precision
    mr = metrics.box.mr  # mean recall
    map50 = metrics.box.map50
    map = metrics.box.map
    print('\n===== Overall (test set) =====')
    print(f'Precision: {mp:.4f}')
    print(f'Recall   : {mr:.4f}')
    print(f'mAP@0.50 : {map50:.4f}')
    print(f'mAP@0.50:0.95: {map:.4f}')

    # ---- 按类别 AP ----
    class_maps = metrics.box.maps  # list: AP per class at IoU 0.50:0.95
    names = model.names
    per_class_rows = []
    print('\n===== Per-class AP (IoU 0.50:0.95) =====')
    for cls_id, ap_c in enumerate(class_maps):
        # ultralytics 里不存在的类别会返回 nan；转为 0 方便观感
        ap_val = 0.0 if ap_c != ap_c else float(ap_c)
        name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        print(f'{cls_id:>2} - {name:<20}: {ap_val:.4f}')
        per_class_rows.append({'class_id': cls_id, 'class_name': name, 'AP_50_95': ap_val})

    # 保存成 CSV 与 JSON
    import csv
    with open(save_dir / 'overall_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Precision', 'Recall', 'mAP50', 'mAP50_95'])
        writer.writeheader()
        writer.writerow({
            'Precision': round(float(mp), 6),
            'Recall': round(float(mr), 6),
            'mAP50': round(float(map50), 6),
            'mAP50_95': round(float(map), 6)
        })

    with open(save_dir / 'per_class_ap.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['class_id', 'class_name', 'AP_50_95'])
        writer.writeheader()
        writer.writerows(per_class_rows)

    # 也存一份原始指标 JSON
    raw = {
        'mp': float(mp),
        'mr': float(mr),
        'map50': float(map50),
        'map50_95': float(map),
        'per_class_ap_50_95': per_class_rows
    }
    with open(save_dir / 'metrics_summary.json', 'w', encoding='utf-8') as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    print(f'\nSaved reports to: {save_dir.resolve()}')
    print('Generated plots: PR curves, confusion matrix, F1/Recall curves in the same folder.')

if __name__ == '__main__':
    main()
