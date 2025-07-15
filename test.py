import os
import random
import cv2
import time
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Union, Any

# Import các hàm tiện ích
from license_plate_utils import normalize_plate, segment_characters

# --- ⚙️ CẤU HÌNH ---
# ✅ Đường dẫn đến các model .tflite đã được tối ưu
YOLO_TFLITE_PATH = r"C:\Users\TriNguyen\Desktop\Code\Python\OpenCV\best416_float32.tflite"
CNN_TFLITE_PATH = r"C:\Users\TriNguyen\Desktop\Code\Python\OpenCV\character_classifier_int8.tflite"

# Các file và thư mục khác
CLASS_NAMES_PATH = 'class_names.txt'
SOURCE_IMAGE_FOLDER = r'C:\Users\TriNguyen\Desktop\Code\Python\OpenCV\data\images'
# SOURCE_IMAGE_FOLDER = r'C:\Users\TriNguyen\Desktop\Code\Python\OpenCV\Missing data'
TARGET_PLATE_WIDTH = 190    
TARGET_PLATE_HEIGHT = 140

# --- CÁC HÀM HELPER CHO TFLITE ---
# --- CÁC HÀM TIỆN ÍCH CHO TFLITE ---

def load_tflite_interpreter(model_path: str) -> Union[tf.lite.Interpreter, None]:
    """Tải một model TFLite và khởi tạo Interpreter."""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=4)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"❌ Lỗi khi tải model TFLite '{model_path}': {e}")
        return None

def run_yolo_tflite(interpreter: tf.lite.Interpreter, image: np.ndarray, conf_threshold=0.6, nms_threshold=0.5) -> List[Tuple[int, int, int, int]]:
    """Chạy YOLO TFLite để phát hiện biển số và trả về các bounding box."""
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    original_h, original_w = image.shape[:2]
    input_h, input_w = input_details['shape'][1:3]

    img_resized = cv2.resize(image, (input_w, input_h))
    input_data = np.expand_dims(img_resized, axis=0)
    
    if input_details['dtype'] == np.float32:
        input_data = (input_data / 255.0).astype(np.float32)
    
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details['index'])[0].T

    boxes, confidences = [], []
    for detection in output_data:
        confidence = detection[4]
        if confidence > conf_threshold:
            cx, cy, w, h = detection[:4]
            x = int((cx - w / 2) * original_w)
            y = int((cy - h / 2) * original_h)
            boxes.append([x, y, int(w * original_w), int(h * original_h)])
            confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            final_boxes.append((x, y, x + w, y + h))
    return final_boxes

def run_cnn_tflite(interpreter: tf.lite.Interpreter, char_image: np.ndarray, class_names: List[str]) -> str:
    """Chạy CNN TFLite trên ảnh ký tự và trả về ký tự dự đoán."""
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    img_resized = cv2.resize(char_image, (input_details['shape'][2], input_details['shape'][1]))
    input_data = np.expand_dims(img_resized, axis=[0, -1])

    if input_details['dtype'] == np.int8:
        input_scale, input_zero_point = input_details['quantization']
        input_data = (input_data.astype(np.float32) / input_scale + input_zero_point).astype(np.int8)
    else:
        input_data = (input_data / 255.0).astype(np.float32)

    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details['index'])
    return class_names[np.argmax(output_data)]

# --- HÀM XỬ LÝ CHÍNH ---

def process_frame(image: np.ndarray, yolo_interpreter: tf.lite.Interpreter, cnn_interpreter: tf.lite.Interpreter, class_names: List[str]) -> Tuple[np.ndarray, float]:
    """
    Hàm xử lý một khung hình và trả về ảnh kết quả cùng thời gian xử lý.
    """
    start_time = time.time()  # Bắt đầu đếm giờ

    boxes = run_yolo_tflite(yolo_interpreter, image)
    
    for (x1, y1, x2, y2) in boxes:
        plate_image = image[y1:y2, x1:x2]
        if plate_image.size == 0:
            continue
            
        normalized_plate, _ = normalize_plate(plate_image)
        char_data_list, _ = segment_characters(normalized_plate, debug=False)
        
        if char_data_list:
            recognized_text = ""
            for char_img, _ in char_data_list:
                predicted_char = run_cnn_tflite(cnn_interpreter, char_img, class_names)
                    
                if predicted_char.lower() != 'noise':
                    recognized_text += predicted_char
            
            # Chỉ vẽ lên ảnh nếu chuỗi nhận dạng không rỗng
            if recognized_text:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, recognized_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    end_time = time.time()  # Kết thúc đếm giờ
    processing_time = end_time - start_time  # Tính thời gian xử lý (giây)
    
    return image, processing_time

# --- CÁC CHẾ ĐỘ CHẠY ---

def run_from_webcam(yolo, cnn, classes):
    """Chạy pipeline từ webcam."""
    print("\n▶️ Chế độ WEBCAM (TFLite). Nhấn 'q' để thoát.")
    cap = cv2.VideoCapture(0) # Thử camera 0 (mặc định) trước
    if not cap.isOpened():
        print("❌ Lỗi: Không thể mở webcam. Thử camera 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
             print("❌ Lỗi: Vẫn không thể mở webcam."); return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_frame(frame, yolo, cnn, classes)
        cv2.imshow('Webcam TFLite Recognition', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def run_from_folder(folder_path, yolo, cnn, classes):
    """Chạy pipeline trên một ảnh ngẫu nhiên từ thư mục."""
    print(f"\n▶️ Chế độ FILE. Chọn ảnh ngẫu nhiên từ '{folder_path}'.")
    if not os.path.isdir(folder_path) or not os.listdir(folder_path):
        print(f"❌ Lỗi: Thư mục không tồn tại hoặc trống: '{folder_path}'")
        return
        
    random_img_name = random.choice(os.listdir(folder_path))
    random_img_path = os.path.join(folder_path, random_img_name)
    print(f"   - Đang xử lý ảnh: {random_img_name}")
    
    image = cv2.imread(random_img_path)
    if image is None:
        print(f"❌ Lỗi: không thể đọc file ảnh '{random_img_path}'")
        return

    processed_image, proc_time = process_frame(image, yolo, cnn, classes)
    
    print(f"   - ⏱️ Thời gian nhận diện: {proc_time * 1000:.2f} ms")
    
    cv2.imshow("File Recognition Result", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- HÀM MAIN ---
def main():
    """Hàm chính: tải model, file nhãn và chạy menu lựa chọn."""
    yolo_interpreter = load_tflite_interpreter(YOLO_TFLITE_PATH)
    cnn_interpreter = load_tflite_interpreter(CNN_TFLITE_PATH)
    
    try:
        with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f]
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file nhãn: {CLASS_NAMES_PATH}"); class_names = None

    if not all([yolo_interpreter, cnn_interpreter, class_names]):
        print("Thoát chương trình do không tải được model hoặc file nhãn.")
        return

    while True:
        print("\n--- CHỌN CHẾ ĐỘ NHẬN DẠNG (TFLite) ---")
        print("  1. Nhận dạng từ Webcam")
        print("  2. Nhận dạng từ File ảnh ngẫu nhiên")
        print("  Nhấn phím khác để thoát.")
        
        choice = input("Nhập lựa chọn của bạn (1 hoặc 2): ")
        
        if choice == '1':
            run_from_webcam(yolo_interpreter, cnn_interpreter, class_names)
        elif choice == '2':
            run_from_folder(SOURCE_IMAGE_FOLDER, yolo_interpreter, cnn_interpreter, class_names)
        else:
            print("Đã thoát chương trình.")
            break

if __name__ == '__main__':
    main()