from ultralytics import YOLO

if __name__ == '__main__':
    # 1. โหลดโมเดล
    model = YOLO("runs/detect/train-2/weights/best.pt")

    # 2. ทำการ Predict (เปลี่ยน source เป็นตำแหน่งภาพจริงของคุณ)
    results = model.predict(
        source="frame/images/0001.jpg",  # <--- แก้ไขจุดนี้ให้ตรงกับรูปที่มีจริง
        conf=0.5,
        save=True
    )

    # 3. แสดงผลลัพธ์
    results[0].show()