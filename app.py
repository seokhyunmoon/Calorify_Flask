from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import csv
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = YOLO('best.pt')  # 또는 여러분이 학습시킨 모델 경로

# 영어 이름을 한국어 이름으로 변환하는 딕셔너리 생성
food_name_mapping = {}
with open('food_names.csv', 'r', encoding='utf-8') as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        food_name_mapping[row['romanized_name'].lower()] = row['korean_name']

def translate_food_name(english_name):
    return food_name_mapping.get(english_name.lower(), english_name)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        results = model(image)
        
        objects = []
        food_classes = set()
        for r in results:
            for box in r.boxes:
                class_name = r.names[int(box.cls)]
                korean_class_name = translate_food_name(class_name)
                confidence = float(box.conf)
                bbox = box.xyxy[0].tolist()
                
                obj = {
                    'class': korean_class_name,
                    'confidence': confidence,
                    'bbox': bbox
                }
                objects.append(obj)
                food_classes.add(korean_class_name)

                # 바운딩 박스 그리기
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(image, f"{korean_class_name}: {confidence:.2f}", (int(bbox[0]), int(bbox[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 이미지를 base64로 인코딩
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return render_template('result.html', 
                               objects=objects, 
                               food_classes=list(food_classes), 
                               image=image_base64)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=5000, debug=True)