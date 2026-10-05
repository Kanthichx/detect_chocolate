import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. แก้ไขพาธโมเดลให้ชี้ไปที่ train-2
    model = YOLO("runs/detect/train-2/weights/best.pt")

    # 2. ทำการ Predict วิดีโอ (ระบุชื่อไฟล์วิดีโอของคุณ เช่น 'train_candy2.mp4' หรือวิดีโอทดสอบอื่น)
    results = model.predict(
        source="test.mp4",  # <--- เปลี่ยนเป็นชื่อไฟล์วิดีโอที่คุณต้องการทดสอบ
        conf=0.5,                   # กำหนดค่า Confidence ขั้นต่ำที่ 50%
        save=True,                  # บันทึกวิดีโอผลลัพธ์ลงใน runs/detect/predict
        show=True                   # แสดงหน้าต่างเล่นวิดีโอขณะประมวลผล
    )