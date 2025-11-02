from ultralytics import YOLO

model = YOLO("models/yolo11n.pt")          # 全局加载一次


def top1_class(img_path: str) -> str:
    """
    返回置信度最高的单个目标类别名，
    若没有任何检测框则返回空字符串 ''。
    """
    r = model(img_path, verbose=False)[0]   # 单图推理
    if r.boxes is None or len(r.boxes) == 0:
        return ''
    # 取置信度最大的索引
    best_idx = r.boxes.conf.cpu().argmax().item()
    cls_id = int(r.boxes.cls[best_idx].item())
    return r.names[cls_id]


# ------------- 示例 -------------
if __name__ == "__main__":
    print(top1_class(r"ultralytics\assets\bus.jpg"))   # 可能输出: bus