# Thai Language Model Implementation

ระบบโมเดลภาษาไทยอย่างง่าย (Simple Thai Language Model) - JavaScript implementation of a transformer-based language model for Thai text processing.

## โครงสร้างของโมเดล (Model Architecture)

โมเดลนี้ประกอบด้วยส่วนประกอบหลักดังนี้:

### การกำหนดค่าโมเดล (Model Configuration)
```javascript
{
  numLayers: 4,        // จำนวน transformer layers
  numHeads: 4,         // จำนวน attention heads
  embedDim: 256,       // ขนาดของ embedding vector
  ffnDim: 512,         // ขนาดของ feed-forward network
  maxSeqLength: 128,   // ความยาวสูงสุดของ sequence
  vocabSize: 128,      // ขนาดของ vocabulary
  dropout: 0.1         // อัตรา dropout
}
```

### ส่วนประกอบหลัก (Main Components)

1. **Positional Encoding**
   - สร้าง position embeddings ด้วย sine และ cosine functions
   - ช่วยให้โมเดลเข้าใจลำดับของคำในประโยค

2. **Multi-Head Self-Attention**
   - คำนวณ attention scores ระหว่างทุกคู่ของ tokens
   - รองรับการทำงานแบบ multi-head เพื่อจับความสัมพันธ์ที่หลากหลาย
   - มีการใช้ causal mask เพื่อป้องกันการมองไปข้างหน้า

3. **Feed-Forward Network**
   - ประกอบด้วย linear transformations สองชั้น
   - ใช้ ReLU activation function
   - ช่วยในการประมวลผลข้อมูลแบบ non-linear

4. **Layer Normalization**
   - ปรับค่าให้อยู่ในช่วงที่เหมาะสม
   - ช่วยให้การเทรนโมเดลมีเสถียรภาพ

### การประมวลผล (Processing Flow)

1. แปลงข้อความเป็น tokens
2. สร้าง embeddings และเพิ่ม positional encoding
3. ส่งผ่าน transformer layers ตามจำนวนที่กำหนด
4. คำนวณ probabilities สำหรับ token ถัดไป

## การใช้งาน (Usage)

```javascript
const input = "สวัสดี! มีอะไรให้ฉันช่วยไหม?";
const output = thaiLanguageModel(input, CONFIG);
```

## ฟังก์ชันสนับสนุน (Helper Functions)

โมเดลมีฟังก์ชันสนับสนุนสำหรับการคำนวณทางคณิตศาสตร์:
- `matrixMultiply`: คูณเมทริกซ์
- `transpose`: สลับแถวและคอลัมน์ของเมทริกซ์
- `softmax`: คำนวณ softmax probabilities
- `createRandomMatrix`: สร้างเมทริกซ์สุ่ม
- `applyDropout`: ใช้ dropout เพื่อป้องกัน overfitting

## ข้อจำกัด (Limitations)

1. ใช้ vocabulary ขนาดเล็ก (128 tokens)
2. รองรับความยาวประโยคสูงสุด 128 tokens
3. ไม่มีการ optimize ประสิทธิภาพการทำงาน
4. ใช้การ tokenize อย่างง่ายด้วย UTF-8

## การพัฒนาต่อ (Future Improvements)

1. เพิ่มขนาด vocabulary
2. ปรับปรุงระบบ tokenization
3. เพิ่มประสิทธิภาพการคำนวณ
4. รองรับการ fine-tuning
5. เพิ่ม pre-training options


![image](https://github.com/user-attachments/assets/32211146-6097-42c9-88f9-c7bcdde981f9)
