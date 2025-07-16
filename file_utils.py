import os
import cv2
import numpy as np
import time
from typing import Dict

# Giả định các file này nằm cùng cấp thư mục
from license_plate_utils import normalize_plate, segment_characters
from libert import load_tflite_interpreter, run_cnn_tflite, run_yolo_tflite

# --- CẤU HÌNH ---
# YOLO_TFLITE_PATH = r"C:\Users\TriNguyen\Desktop\Code\Python\OpenCV\best_float32.tflite"
YOLO_TFLITE_PATH = "best416_float32.tflite"
CNN_TFLITE_PATH = "character_classifier_int8.tflite"

# Các file và thư mục khác
CLASS_NAMES_PATH = 'class_names.txt'
SOURCE_IMAGE_FOLDER = 'test_img'

def benchmark_pipeline(image: np.ndarray, yolo, cnn, classes) -> Dict[str, float]:
    """
    Chạy toàn bộ pipeline và trả về một dictionary chứa thời gian của từng bước.
    """
    timings = {
        'yolo_detect': 0.0,
        'plate_normalize': 0.0,
        'char_segment': 0.0,
        'char_recognize': 0.0,
        'total': 0.0
    }
    
    overall_start_time = time.time()

    # 1. Phát hiện biển số (YOLO)
    t0 = time.time()
    boxes = run_yolo_tflite(yolo, image)
    t1 = time.time()
    timings['yolo_detect'] = t1 - t0
    
    # Xử lý cho từng biển số được phát hiện
    for (x1, y1, x2, y2) in boxes:
        plate_image = image[y1:y2, x1:x2]
        if plate_image.size == 0:
            continue
            
        # 2. Chuẩn hóa biển số
        t2 = time.time()
        normalized_plate, _ = normalize_plate(plate_image)
        t3 = time.time()
        timings['plate_normalize'] += (t3 - t2)
        
        # 3. Phân đoạn ký tự
        t4 = time.time()
        char_data_list, _ = segment_characters(normalized_plate)
        t5 = time.time()
        timings['char_segment'] += (t5 - t4)
        
        # 4. Nhận dạng từng ký tự (CNN)
        if char_data_list:
            cnn_time_for_plate = 0.0
            for char_img, _ in char_data_list:
                t6 = time.time()
                _ = run_cnn_tflite(cnn, char_img, classes)
                t7 = time.time()
                cnn_time_for_plate += (t7 - t6)
            timings['char_recognize'] += cnn_time_for_plate
    
    timings['total'] = time.time() - overall_start_time
    return timings

def analyze_component_performance():
    """
    Hàm chính để chạy phân tích và in ra kết quả chi tiết từng thành phần.
    """
    print("--- BẮT ĐẦU PHÂN TÍCH HIỆU NĂNG TỪNG THÀNH PHẦN ---")

    # 1. Tải model và tài nguyên
    print("⏳ Đang tải models...")
    yolo_interpreter = load_tflite_interpreter(YOLO_TFLITE_PATH)
    cnn_interpreter = load_tflite_interpreter(CNN_TFLITE_PATH)
    try:
        with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f]
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file nhãn: {CLASS_NAMES_PATH}"); return

    if not all([yolo_interpreter, cnn_interpreter, class_names]):
        print("❌ Thoát do không tải được model hoặc file nhãn."); return
    print("✅ Tải model thành công.")

    # 2. Lấy danh sách ảnh
    image_files = [f for f in os.listdir(SOURCE_IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"❌ Lỗi: Không tìm thấy ảnh trong '{SOURCE_IMAGE_FOLDER}'."); return
    
    total_images = len(image_files)
    print(f"🔎 Tìm thấy {total_images} ảnh để phân tích.\n")

    # 3. Chạy và thu thập dữ liệu
    all_timings = []
    for i, filename in enumerate(image_files):
        print(f"   - Đang xử lý ảnh {i + 1}/{total_images}: {filename}", end='\r')
        image_path = os.path.join(SOURCE_IMAGE_FOLDER, filename)
        image = cv2.imread(image_path)
        if image is None: continue
        
        timings = benchmark_pipeline(image, yolo_interpreter, cnn_interpreter, class_names)
        all_timings.append(timings)
    
    print("\n\n✅ Đã xử lý xong tất cả các ảnh.")

    # 4. Phân tích và in kết quả
    if not all_timings:
        print("Không có ảnh nào được xử lý."); return

    # Tính tổng thời gian cho mỗi thành phần trên tất cả các lần chạy
    avg_timings = {key: 0.0 for key in all_timings[0]}
    for timings in all_timings:
        for key in timings:
            avg_timings[key] += timings[key]
    
    # Lấy trung bình
    num_runs = len(all_timings)
    for key in avg_timings:
        avg_timings[key] /= num_runs

    # In kết quả
    total_avg_time_ms = avg_timings['total'] * 1000
    print("\n---📊 KẾT QUẢ PHÂN TÍCH TỪNG THÀNH PHẦN 📊---")
    print("----------------------------------------------------------------")
    print(f"Thời gian xử lý trung bình tổng thể: {total_avg_time_ms:.2f} ms/ảnh\n")
    
    print(f"{'Thành phần':<25} | {'Thời gian (ms)':<15} | {'Tỷ trọng (%)'}")
    print("----------------------------------------------------------------")
    
    for key, avg_time_sec in avg_timings.items():
        if key == 'total': continue
        
        avg_time_ms = avg_time_sec * 1000
        percentage = (avg_time_sec / avg_timings['total']) * 100 if avg_timings['total'] > 0 else 0
        
        print(f"{key:<25} | {avg_time_ms:<15.2f} | {percentage:.2f}%")
        
    print("----------------------------------------------------------------\n")


if __name__ == '__main__':
    print("Bắt đầu chạy chương trình")
    analyze_component_performance()
