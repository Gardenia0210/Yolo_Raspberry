from ultralytics import YOLO
from collections import Counter
import cv2

# 全局加载一次模型，避免重复 I/O
model = YOLO("models/yolo11n.pt")   # 换成你自己的权重路径


def predict_summary(img_path: str) -> str:
    """
    对单张图片做 YOLO 推理，返回形如
    '4 persons, 1 bus' 的字符串。
    若没检测到任何目标，返回 'no detections'。
    """
    # 推理
    results = model(img_path, verbose=False)  # 返回列表，单张图取第 0 个
    r = results[0]

    # 汇总类别
    names = r.names            # {0: 'person', 1: 'bicycle', ...}
    cls_idx = r.boxes.cls.cpu().int().tolist()  # 所有框的类别索引
    if not cls_idx:            # 空列表
        return "no detections"

    counter = Counter(cls_idx)
    # 按字母序拼字符串，方便保持一致
    parts = [f"{cnt} {names[c]}" for c, cnt in sorted(counter.items())]
    return ", ".join(parts)


# ----------------- 使用示例 -----------------
if __name__ == "__main__":
    print(predict_summary(r"ultralytics\assets\bus.jpg"))#改为图片路径