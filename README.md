# 🍫 Chocolate Detection and Classification Using YOLO26

โปรเจกต์สำหรับพัฒนาโมเดลปัญญาประดิษฐ์ด้วย **YOLO26** เพื่อตรวจจับและจำแนกช็อกโกแลต 5 คลาสจากภาพ วิดีโอ และกล้อง Webcam แบบ Real-time โดยใช้ **Ultralytics YOLO** ร่วมกับ **Label Studio** สำหรับกำหนด Bounding Box และเตรียม Dataset

ขั้นตอนหลักของโปรเจกต์ ได้แก่ การเตรียมภาพ การทำ Label การแปลงข้อมูลเป็นรูปแบบ YOLO การ Train โมเดล และการทดสอบผลลัพธ์


---

## 📂 Project Structure

ไฟล์หลักที่มีอยู่ในโปรเจกต์:

```text
detect_chocolate-main/
├── README.md
├── .gitignore
├── requirements.txt
├── 01-export_dataset.py
├── 02-train.py
├── 03-test_image.py
├── 04-test_video.py
├── 05-test-camera.py
├── project-1-at-2026-10-05-21-40-5d7a3a3c.json
└── dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    ├── classes.txt
    └── data.yaml
```

โฟลเดอร์และไฟล์ที่ต้องเตรียมหรือจะเกิดขึ้นเมื่อใช้งาน:

```text
detect_chocolate-main/
├── env/                 # Virtual Environment ที่สร้างเอง
├── Frame/
│   └── images/          # ภาพต้นฉบับสำหรับทำ Label และ Export Dataset
├── images/              # ภาพประกอบ README ที่แคปเอง
├── train_chocolate.mp4  # ตัวอย่างชื่อวิดีโอต้นฉบับ ต้องเพิ่มเอง
├── test.mp4             # วิดีโอทดสอบ ต้องเพิ่มเอง
└── runs/
    └── detect/          # ผลลัพธ์จากการ Train และ Predict
```

> ZIP มี Dataset และไฟล์ JSON แล้ว แต่ไม่มีโฟลเดอร์ภาพต้นฉบับ `Frame/images`, วิดีโอทดสอบ หรือไฟล์โมเดลที่เทรนแล้ว `best.pt` หากใช้ Dataset ที่มีอยู่ สามารถข้ามขั้นตอนทำ Label และแปลง JSON แล้วเริ่มจากแก้ `dataset/data.yaml` ก่อน Train ได้เลย

> **📸 ใส่รูปที่ 1:** แคปหน้า VS Code ที่แสดงโครงสร้างไฟล์โปรเจกต์ บันทึกเป็น `images/01-project-structure.png`

<!-- ![โครงสร้างโปรเจกต์](images/01-project-structure.png) -->

---

# 🏷️ Image Labeling

โปรเจกต์นี้ใช้ **Label Studio** สำหรับลาก Bounding Box ครอบช็อกโกแลตและกำหนด Class ก่อนนำข้อมูลไป Train โมเดล

ข้อมูล Label ที่ส่งออกเป็น JSON จะถูกแปลงด้วย `01-export_dataset.py` เป็นไฟล์ `.txt` รูปแบบ YOLO พร้อมแบ่งภาพเป็น Training และ Validation

---

# ⚙️ Installation

## 1. เปิดโฟลเดอร์โปรเจกต์

แตกไฟล์ ZIP แล้วเปิด Command Prompt ในโฟลเดอร์ที่มี `02-train.py` และ `dataset` อยู่ คำสั่งต่อไปนี้ใช้สำหรับ **Windows Command Prompt (CMD)**

```cmd
cd /d "C:\path\to\detect_chocolate-main"
```

เปลี่ยน `C:\path\to\detect_chocolate-main` เป็นตำแหน่งจริงในเครื่อง

## 2. สร้างและ Activate Virtual Environment

```cmd
py -3 -m venv env
env\Scripts\activate.bat
```

เมื่อ Activate สำเร็จจะเห็น `(env)` อยู่หน้าบรรทัดคำสั่ง

หากใช้ PowerShell ให้ Activate ด้วย:

```powershell
.\env\Scripts\Activate.ps1
```

> **📸 ใส่รูปที่ 2:** แคป Terminal หลัง Activate สำเร็จ โดยให้เห็น `(env)` บันทึกเป็น `images/02-activate-env.png`

<!-- ![Activate Virtual Environment](images/02-activate-env.png) -->

## 3. ติดตั้ง Dependencies

```cmd
python -m pip install --upgrade pip
pip install -U ultralytics
pip install opencv-python label-studio
```

`ultralytics` ใช้สำหรับ Train และ Predict, `opencv-python` ใช้สำหรับกล้องและแสดงภาพ ส่วน `label-studio` ใช้เตรียม Label หากใช้ Dataset ที่มีอยู่แล้ว ไม่จำเป็นต้องทำขั้นตอน Label ใหม่

> `requirements.txt` ใน ZIP ยังเป็นไฟล์ว่าง จึงต้องติดตั้ง Package ตามคำสั่งด้านบน

## 4. ตรวจสอบอุปกรณ์สำหรับ Train

ใน `02-train.py` และ `05-test-camera.py` กำหนด `device=0` เพื่อใช้ GPU ตัวแรก ตรวจสอบว่า PyTorch มองเห็น CUDA ได้ด้วย:

```cmd
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

หากใช้ CPU ให้เปลี่ยนค่าในทั้งสองไฟล์เป็น:

```python
device="cpu"
```

> **📸 ใส่รูปที่ 3:** แคปผลติดตั้ง Package และผลตรวจสอบ CUDA บันทึกเป็น `images/03-installation.png`

<!-- ![ติดตั้ง Dependencies และตรวจสอบ CUDA](images/03-installation.png) -->

---

# 🎞️ Extract Frame จาก Video

ขั้นตอนนี้ใช้เมื่อต้องการสร้าง Dataset จากวิดีโอใหม่ ต้องเตรียมไฟล์วิดีโอเอง ตัวอย่างนี้ตั้งชื่อว่า `train_chocolate.mp4` ซึ่งไม่ได้รวมอยู่ใน ZIP

หลังติดตั้ง FFmpeg และเรียกคำสั่ง `ffmpeg` ได้แล้ว ให้สร้างโฟลเดอร์เก็บภาพ:

```cmd
mkdir Frame\images
```

แยกภาพจากวิดีโอ 2 เฟรมต่อวินาที:

```cmd
ffmpeg -i train_chocolate.mp4 -vf fps=2 Frame/images/%04d.jpg
```

| ส่วนของคำสั่ง | ความหมาย |
| --- | --- |
| `ffmpeg -i` | ระบุไฟล์วิดีโอที่นำมาแยกเฟรม |
| `train_chocolate.mp4` | ชื่อวิดีโอตัวอย่าง เปลี่ยนให้ตรงกับไฟล์จริง |
| `-vf fps=2` | ดึงภาพ 2 เฟรมต่อวินาที |
| `Frame/images/%04d.jpg` | เก็บภาพเป็น `0001.jpg`, `0002.jpg`, … |

ค่า `fps` ปรับได้ตามจำนวนภาพที่ต้องการ ใช้ชื่อโฟลเดอร์ `Frame` ให้ตรงกับ `IMAGES_DIR` ในสคริปต์ Export

> **📸 ใส่รูปที่ 4:** แคปคำสั่งแยกเฟรมและภาพที่ได้ในโฟลเดอร์ `Frame/images` บันทึกเป็น `images/04-extract-frames.png`

<!-- ![แยกเฟรมจากวิดีโอ](images/04-extract-frames.png) -->

---

# 🖼️ เริ่ม Label ด้วย Label Studio

เปิด CMD อีกหนึ่งหน้าต่าง เข้าโฟลเดอร์โปรเจกต์ และ Activate Environment จากนั้นตั้งค่าสำหรับเปิดภาพจาก Local Storage:

```cmd
env\Scripts\activate.bat
set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=C:\path\to\detect_chocolate-main
label-studio start
```

เปลี่ยน Path ให้ตรงกับโฟลเดอร์โปรเจกต์จริง แล้วเข้า URL ที่แสดงใน Terminal เพื่อใช้งาน Label Studio โดยเปิด CMD หน้าต่างนี้ค้างไว้ระหว่างทำงาน

หากเป็นการใช้งานครั้งแรก ให้สร้างบัญชีสำหรับเข้าใช้งานระบบในเครื่อง

> **📸 ใส่รูปที่ 5:** แคป CMD ที่รัน Label Studio และหน้าเว็บที่เปิดสำเร็จ บันทึกเป็น `images/05-label-studio-start.png`

<!-- ![เปิด Label Studio](images/05-label-studio-start.png) -->

---

# 🏷️ ตั้งค่า Labeling

## 1. สร้าง Project

กด **Create Project** แล้วตั้งชื่อ เช่น `Chocolate Detection` จากนั้นเลือก Labeling Setup แบบ **Object Detection with Bounding Boxes**

> **📸 ใส่รูปที่ 6:** แคปหน้าสร้าง Project และการเลือก Object Detection with Bounding Boxes บันทึกเป็น `images/06-create-project.png`

<!-- ![สร้าง Project สำหรับตรวจจับช็อกโกแลต](images/06-create-project.png) -->

## 2. นำเข้าภาพ

หากใช้ Local Storage ให้เข้า **Settings → Cloud Storage** แล้วเพิ่ม **Local Files** ระบุ Absolute Local Path เป็นโฟลเดอร์ภาพ เช่น:

```text
C:\path\to\detect_chocolate-main\Frame\images
```

ทดสอบการเชื่อมต่อ ตรวจสอบตัวอย่างไฟล์ และ Sync ภาพเข้ามาเป็น Tasks โดยเลือกให้นำเข้าไฟล์ภาพแต่ละไฟล์เป็น Task

สามารถนำเข้าภาพผ่านหน้า Data Import ได้เช่นกัน แต่ก่อนแปลง Dataset ต้องมีภาพต้นฉบับใน `Frame/images` และตรวจสอบชื่อไฟล์ให้สัมพันธ์กับ JSON ที่ Export

> **📸 ใส่รูปที่ 7:** แคปการตั้งค่า Local Storage โดยให้เห็น Path ของภาพ บันทึกเป็น `images/07-local-storage.png`

<!-- ![ตั้งค่า Local Storage](images/07-local-storage.png) -->

> **📸 ใส่รูปที่ 8:** แคปหน้า Project หลังนำเข้าภาพแล้ว บันทึกเป็น `images/08-imported-images.png`

<!-- ![ภาพที่นำเข้าสำหรับทำ Label](images/08-imported-images.png) -->

## 3. Classes

ชื่อคลาสต่อไปนี้มาจาก `dataset/classes.txt` และ `dataset/data.yaml` โดยคงการสะกดตามไฟล์จริง:

| Class ID | ชื่อ Class |
| --- | --- |
| 0 | `chocalateHershey` |
| 1 | `chocolateBonBon` |
| 2 | `chocolateCookkies` |
| 3 | `chocolateGeentea` |
| 4 | `chocoleteBengBeng` |

> ชื่อ Class ต้องตรงกันระหว่าง Label, Dataset และโมเดล หากเปลี่ยนชื่อ ต้องปรับข้อมูลที่เกี่ยวข้องให้สอดคล้องกันด้วย

เข้า **Settings → Labeling Interface** เลือกโหมด Code แล้วใช้ตัวอย่างนี้เพื่อกำหนดคลาสให้ตรงกับ Dataset:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="chocalateHershey" background="#9F0909"/>
    <Label value="chocolateBonBon" background="#FFA39E"/>
    <Label value="chocolateCookkies" background="#AD8B00"/>
    <Label value="chocolateGeentea" background="#389E0D"/>
    <Label value="chocoleteBengBeng" background="#007EBD"/>
  </RectangleLabels>
</View>
```

กด Save แล้วกลับไปหน้า Project เพื่อเริ่มทำ Label

> **📸 ใส่รูปที่ 9:** แคปหน้า Labeling Interface ที่มีครบทั้ง 5 คลาส บันทึกเป็น `images/09-labeling-interface.png`

<!-- ![Labeling Interface และคลาสช็อกโกแลต](images/09-labeling-interface.png) -->

---

# ✏️ ทำ Bounding Box

1. เปิดภาพที่ต้องการทำ Label
2. เลือก Class ให้ตรงกับช็อกโกแลตในภาพ
3. ลาก Bounding Box ครอบวัตถุแต่ละชิ้น
4. ตรวจสอบกรอบและชื่อ Class
5. กด **Submit** แล้วทำภาพถัดไป

ควรครอบวัตถุให้พอดีและกำหนดคลาสอย่างสม่ำเสมอ เพื่อให้ข้อมูลเหมาะสำหรับ Train

> **📸 ใส่รูปที่ 10:** แคปภาพที่ลาก Bounding Box และกำหนดชื่อคลาสเรียบร้อยแล้ว บันทึกเป็น `images/10-bounding-box.png`

<!-- ![ตัวอย่าง Bounding Box ของช็อกโกแลต](images/10-bounding-box.png) -->

---

# 📤 Export Annotation

หลังทำ Label เสร็จ ให้กลับไปหน้า Project กด **Export** เลือกรูปแบบ **JSON** แล้วดาวน์โหลดไฟล์มาไว้ข้าง `01-export_dataset.py`

ไฟล์ JSON ที่มีอยู่ใน ZIP คือ:

```text
project-1-at-2026-10-05-21-40-5d7a3a3c.json
```

> สคริปต์จะเลือกไฟล์ `.json` ตัวแรกเมื่อเรียงชื่อในโฟลเดอร์ หากมีหลายไฟล์ ให้ตรวจสอบข้อความ `Using JSON file` ว่าเลือกไฟล์ที่ต้องการ

> **📸 ใส่รูปที่ 11:** แคปหน้าต่าง Export ที่เลือกรูปแบบ JSON บันทึกเป็น `images/11-export-json.png`

<!-- ![Export Annotation เป็น JSON](images/11-export-json.png) -->

---

# 🔄 Convert Label Studio JSON → YOLO Dataset

## 1. ตรวจสอบ Path ในสคริปต์

เปิด `01-export_dataset.py` และตรวจสอบค่า:

```python
SCRIPT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = SCRIPT_DIR / "Frame" / "images"
OUTPUT_DIR = SCRIPT_DIR / "dataset"
TRAIN_SPLIT = 0.8
SEED = 42
```

| ตัวแปร | ความหมาย |
| --- | --- |
| `IMAGES_DIR` | โฟลเดอร์ภาพต้นฉบับที่ตรงกับภาพใน JSON |
| `OUTPUT_DIR` | โฟลเดอร์ปลายทางสำหรับ Dataset |
| `TRAIN_SPLIT = 0.8` | แบ่งข้อมูลสำหรับ Train 80% และ Validation 20% |
| `SEED = 42` | กำหนดค่าเริ่มต้นของการสุ่มแบ่งข้อมูล |

สคริปต์อ่านชื่อคลาสจาก Bounding Box ใน JSON และเรียงชื่อก่อนกำหนด Class ID จากนั้นสร้างไฟล์ Label รูปแบบ:

```text
class_id x_center y_center width height
```

ค่าพิกัดและขนาดถูกปรับให้อยู่ในสัดส่วนของภาพ

> ZIP ไม่มีภาพต้นฉบับใน `Frame/images` จึงต้องเพิ่มภาพที่ตรงกับ JSON ก่อนรันขั้นตอนนี้ หากต้องการ Train จาก Dataset ที่ให้มาแล้ว ให้ข้ามการแปลง JSON

## 2. Run Dataset Conversion

```cmd
python 01-export_dataset.py
```

เมื่อสำเร็จจะได้ `dataset/images`, `dataset/labels`, `dataset/classes.txt` และ `dataset/data.yaml` พร้อมข้อความจำนวนภาพ Train และ Validation

> **📸 ใส่รูปที่ 12:** แคป Terminal หลังแปลง Dataset สำเร็จ ให้เห็นชื่อคลาสและจำนวนภาพ บันทึกเป็น `images/12-export-dataset.png`

<!-- ![ผลการแปลง Dataset](images/12-export-dataset.png) -->

## 3. Dataset ที่มีอยู่ใน ZIP

| ชุดข้อมูล | จำนวนภาพ |
| --- | ---: |
| Training | 165 |
| Validation | 42 |
| รวม | 207 |

จำนวนนี้นับจากไฟล์ภาพใน Dataset ที่แนบมา ผลการแปลงใหม่อาจต่างกันตามภาพต้นฉบับและ Annotation ที่ใช้งาน

หากใช้ Dataset เดิม ให้แก้ `path` ใน `dataset/data.yaml` เพราะไฟล์ที่แนบมายังอ้างถึง `C:/Users/Admin/detect_chocolate/dataset` ตัวอย่างค่าหลังแก้:

```yaml
path: C:/path/to/detect_chocolate-main/dataset
train: images/train
val: images/val

nc: 5
names: ['chocalateHershey', 'chocolateBonBon', 'chocolateCookkies', 'chocolateGeentea', 'chocoleteBengBeng']
```

เปลี่ยน `path` ให้เป็นตำแหน่ง Dataset จริง โดยใช้ `/` ตามตัวอย่าง

> **📸 ใส่รูปที่ 13:** แคป `dataset/data.yaml` หลังแก้ Path และโครงสร้าง Dataset บันทึกเป็น `images/13-dataset-config.png`

<!-- ![ตั้งค่า Dataset](images/13-dataset-config.png) -->

---

# 🧠 Train YOLO26

เปิดไฟล์ `02-train.py` ซึ่งโหลดโมเดลเริ่มต้น:

```python
model = YOLO('yolo26n.pt')
```

ค่าที่กำหนดไว้ในสคริปต์:

| Parameter | ค่า |
| --- | --- |
| Dataset | `dataset/data.yaml` |
| Epochs | `100` |
| Image Size | `640` |
| Optimizer | `MuSGD` |
| Device | `0` |

Data Augmentation ที่ใช้:

```python
degrees=15.0
shear=5.0
perspective=0.001
fliplr=0.5
flipud=0.0
mosaic=1.0
mixup=0.1
close_mosaic=10
```

ปรับค่าได้ในไฟล์ Training โดยตรวจสอบ `device` และ Dataset Path ก่อนรัน

## Run Training

```cmd
python 02-train.py
```

> **📸 ใส่รูปที่ 14:** แคป Terminal ระหว่าง Train ให้เห็นโมเดลและค่า Epoch บันทึกเป็น `images/14-training.png`

<!-- ![การ Train YOLO26](images/14-training.png) -->

เมื่อ Train เสร็จ ให้ดูตำแหน่งผลลัพธ์จาก Terminal และหาไฟล์ `weights/best.pt` ในโฟลเดอร์ของรอบนั้น แล้วนำ Path ไปใส่ในสคริปต์ทดสอบทั้งสามไฟล์

> ในไฟล์ทดสอบที่แนบมาใช้ `runs/detect/train-2/weights/best.pt` แต่ ZIP ไม่มีไฟล์นี้ ต้องแก้ให้ตรงกับผลการ Train จริงของเครื่อง

> **📸 ใส่รูปที่ 15:** แคป Terminal หลัง Train เสร็จ ให้เห็นผล Validation และตำแหน่งบันทึกผล บันทึกเป็น `images/15-training-complete.png`

<!-- ![ผลหลัง Train เสร็จ](images/15-training-complete.png) -->

> **📸 ใส่รูปที่ 16:** ใส่กราฟผล Training จากรอบที่ใช้งาน เช่น `results.png` หากมี บันทึกเป็น `images/16-training-results.png`

<!-- ![กราฟผล Training](images/16-training-results.png) -->

---

# 🧪 Test Model

## 1. Test Image

เปิด `03-test_image.py` แล้วแก้ Path โมเดลและภาพให้ตรงกับไฟล์จริง ค่าที่อยู่ในไฟล์เดิมคือ:

```python
model = YOLO("runs/detect/train-2/weights/best.pt")
results = model.predict(
    source="frame/images/0001.jpg",
    conf=0.5,
    save=True
)
results[0].show()
```

หากใช้โฟลเดอร์ที่เตรียมตาม README นี้ ให้แก้ `source` เป็น `Frame/images/0001.jpg` หรือระบุ Path ของภาพทดสอบอื่นที่มีอยู่จริง

รันคำสั่ง:

```cmd
python 03-test_image.py
```

โปรแกรมจะแสดงภาพผลลัพธ์และบันทึกภาพตามตำแหน่งที่ Ultralytics แจ้งใน Terminal โดยกำหนด Confidence ขั้นต่ำไว้ที่ `0.5`

> **📸 ใส่รูปที่ 17:** แคป Terminal หลังทดสอบภาพ ให้เห็นตำแหน่งบันทึกผล บันทึกเป็น `images/17-image-test-terminal.png`

<!-- ![ผลการรันทดสอบภาพ](images/17-image-test-terminal.png) -->

> **📸 ใส่รูปที่ 18:** ใส่ภาพผลตรวจจับช็อกโกแลต ให้เห็น Bounding Box ชื่อคลาส และ Confidence บันทึกเป็น `images/18-image-result.png`

<!-- ![ผลตรวจจับช็อกโกแลตจากภาพ](images/18-image-result.png) -->

---

# 🎥 2. Test Video

เปิด `04-test_video.py` และแก้ Path โมเดลให้ตรงกับผล Training พร้อมเตรียมไฟล์วิดีโอทดสอบ ค่าในไฟล์เดิมคือ:

```python
model = YOLO("runs/detect/train-2/weights/best.pt")
results = model.predict(
    source="test.mp4",
    conf=0.5,
    save=True,
    show=True
)
```

นำ `test.mp4` มาไว้ในโฟลเดอร์โปรเจกต์ หรือแก้ `source` เป็นไฟล์ที่ต้องการทดสอบ แล้วรัน:

```cmd
python 04-test_video.py
```

โปรแกรมจะแสดงวิดีโอระหว่างตรวจจับและบันทึกผลลัพธ์ ให้ดูโฟลเดอร์ปลายทางจากข้อความใน Terminal เพราะชื่อโฟลเดอร์ Predict อาจต่างกันในแต่ละรอบ

> **📸 ใส่รูปที่ 19:** แคป Terminal ขณะทดสอบวิดีโอ บันทึกเป็น `images/19-video-test-terminal.png`

<!-- ![การทดสอบวิดีโอ](images/19-video-test-terminal.png) -->

> **📸 ใส่รูปที่ 20:** แคปเฟรมจากวิดีโอผลลัพธ์ที่เห็นกรอบตรวจจับและชื่อคลาส บันทึกเป็น `images/20-video-result.png`

<!-- ![ผลตรวจจับช็อกโกแลตจากวิดีโอ](images/20-video-result.png) -->

---

# 📷 3. Test Camera

ใช้ `05-test-camera.py` สำหรับตรวจจับผ่าน Webcam แบบ Real-time ก่อนรันให้แก้ Path ของ `best.pt` ให้ตรงกับโมเดลที่ Train แล้ว

การตั้งค่าหลักในสคริปต์:

```python
cap = cv2.VideoCapture(0)
```

`0` คือกล้องตัวแรก หากใช้กล้องอื่นให้ปรับหมายเลขตามอุปกรณ์

```python
results = model.predict(
    source=frame,
    stream=True,
    conf=0.5,
    device=0
)
```

หากใช้ CPU ให้แก้ `device=0` เป็น `device="cpu"` แล้วรัน:

```cmd
python 05-test-camera.py
```

ระบบจะแสดงภาพจากกล้องพร้อม Bounding Box และชื่อคลาส กด **`q` บนหน้าต่างวิดีโอ** เพื่อออกจากโปรแกรม

> **📸 ใส่รูปที่ 21:** แคปหน้าต่างกล้องขณะตรวจจับช็อกโกแลตแบบ Real-time บันทึกเป็น `images/21-camera-result.png`

<!-- ![ผลตรวจจับช็อกโกแลตผ่าน Webcam](images/21-camera-result.png) -->

---

# ⚠️ Notes

* รันคำสั่งจากโฟลเดอร์หลักของโปรเจกต์ เพื่อให้ Relative Path ในสคริปต์ชี้ถูกตำแหน่ง
* แก้ `path` ใน `dataset/data.yaml` ให้ตรงกับเครื่องก่อน Train จาก Dataset เดิม
* สคริปต์ Export ใช้ `Frame/images` ส่วนสคริปต์ทดสอบภาพเดิมใช้ `frame/images` ให้ปรับชื่อ Path ให้ตรงกับโฟลเดอร์จริง
* การแปลง JSON ต้องมีภาพต้นฉบับที่สัมพันธ์กับชื่อภาพใน JSON หากหาไฟล์ไม่พบ สคริปต์จะข้าม Task นั้น
* `01-export_dataset.py` เขียนข้อมูลลง `dataset` แต่ไม่ได้ล้างไฟล์เก่า หากสร้าง Dataset ใหม่ควรใช้โฟลเดอร์ปลายทางใหม่เพื่อไม่ให้ปนกับข้อมูลเดิม
* ใช้ชื่อคลาสตาม `classes.txt` และ `data.yaml` ให้ตรงกัน รวมถึงการสะกดและลำดับ Class ID
* ZIP ไม่มี `best.pt` จึงต้อง Train โมเดลหรือเตรียมโมเดลของโปรเจกต์ก่อนทดสอบ
* แก้ Path โมเดลใน `03-test_image.py`, `04-test_video.py` และ `05-test-camera.py` ให้ตรงกับไฟล์ที่ได้จริง
* ZIP ไม่มีวิดีโอ `test.mp4` ต้องเพิ่มไฟล์เองหรือแก้ชื่อวิดีโอในสคริปต์
* ค่าผลประเมินและภาพผลลัพธ์ให้ใช้จากการรันจริงของโปรเจกต์ ยังไม่มีผล Training แนบมาใน ZIP


