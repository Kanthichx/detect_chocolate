from ultralytics import YOLO

if __name__ == '__main__':

    # โหลดโมเดล
    model = YOLO('yolo26n.pt')

    # เริ่มเทรนพร้อมตั้งค่า Data Augmentation
    results = model.train(
        data='dataset/data.yaml',  # แก้ไขพาธไฟล์คอนฟิกให้ถูกต้อง
        epochs=200,
        imgsz=640,
        optimizer="MuSGD",
        device=0,

        # --- พารามิเตอร์สำหรับ Data Augmentation หลัก ---
        degrees=15.0,        # สุ่มหมุนภาพ -15 ถึง +15 องศา
        shear=5.0,           # สุ่มบิดภาพแบบเฉียงด้านขนาน
        perspective=0.001,   # เพิ่มความลึก

        # --- การพลิกภาพ (Flip) ---
        fliplr=0.5,          # โอกาส 50% ที่จะพลิกภาพซ้าย-ขวา
        flipud=0.0,          # พลิกภาพบน-ล่าง

        # --- พารามิเตอร์ขั้นสูง ---
        mosaic=1.0,          # รวม 4 ภาพ
        mixup=0.1,           # ซ้อนภาพโปร่งแสง
        close_mosaic=10      # ปิด Mosaic ใน 10 epochs สุดท้าย
    )