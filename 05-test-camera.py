import cv2
from ultralytics import YOLO


def main():
    # 1. แก้ไขพาธโมเดลให้ชี้ไปที่ไฟล์ผลลัพธ์การเทรนล่าสุด (train-2)
    model = YOLO("runs/detect/train-2/weights/best.pt")

    # 2. เปิดใช้งานกล้องเว็บแคม (เลข 0 หมายถึงกล้องตัวแรกของเครื่อง)
    cap = cv2.VideoCapture(0)

    # ตรวจสอบว่าสามารถเปิดกล้องได้หรือไม่
    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return

    # แสดงคำแนะนำการใช้งาน
    print("กด 'q' บนหน้าต่างวิดีโอเพื่อออกจากโปรแกรม")

    # เริ่มการตรวจจับแบบ Real-time
    while True:
        # อ่านภาพจากกล้องทีละเฟรม
        success, frame = cap.read()

        if success:
            # นำภาพจากกล้องเข้าสู่โมเดล YOLO
            # ปรับ device=0 เพื่อใช้ GPU (RTX 3050) ในการประมวลผลให้เฟรมเรทสูงขึ้น
            results = model.predict(
                source=frame,
                stream=True,    # ประมวลผลแบบต่อเนื่อง
                conf=0.5,       # กำหนดค่า Confidence ขั้นต่ำที่ 50%
                device=0        # ใช้ GPU (หากมีปัญหาให้เปลี่ยนกลับเป็น 'cpu')
            )

            # วนลูปเพื่อรับผลลัพธ์จากโมเดล
            for r in results:
                # วาดกรอบ Bounding Box และชื่อ Class ลงบนภาพ
                annotated_frame = r.plot()

            # แสดงภาพที่ผ่านการตรวจจับบนหน้าต่าง
            cv2.imshow("YOLO Real-time Detection", annotated_frame)

            # กดปุ่ม 'q' เพื่อออกจากโปรแกรม
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # หากไม่สามารถอ่านภาพจากกล้องได้ ให้หยุดการทำงาน
        else:
            print("ไม่สามารถอ่านข้อมูลจากกล้องได้")
            break

    # ปิดการเชื่อมต่อกับกล้องและปิดหน้าต่างทั้งหมด
    cap.release()
    cv2.destroyAllWindows()


# เรียกใช้งานฟังก์ชัน main()
if __name__ == '__main__':
    main()