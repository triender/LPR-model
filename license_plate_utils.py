import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Any, Union

# --- HÀM TRỢ GIÚP (HELPER FUNCTION) ---

def _create_debug_plot(images: List[np.ndarray], titles: List[str], main_title: str, grid_shape: Tuple[int, int]) -> np.ndarray:
    """Tạo một ảnh duy nhất chứa nhiều bước xử lý để debug."""
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(15, 8))
    plt.suptitle(main_title, fontsize=16)
    axes = axes.flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    # Tắt các subplot không sử dụng
    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    debug_image_rgb = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    debug_image_bgr = cv2.cvtColor(debug_image_rgb, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return debug_image_bgr

# --- CÁC HÀM TIỆN ÍCH CHÍNH ---

def normalize_plate(image: np.ndarray, width: int = 190, height: int = 140, debug: bool = False) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Cố gắng làm phẳng ảnh biển số bằng Perspective Transform.
    Nếu thất bại, trả về ảnh gốc đã được resize.
    """
    warped_image = cv2.resize(image, (width, height))
    debug_steps = {'images': [], 'titles': []}

    if debug:
        debug_steps['images'].append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        debug_steps['titles'].append('1. Original Cropped')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_edges = cv2.Canny(blurred, 100, 200)
    dilated_edges = cv2.dilate(canny_edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    
    if debug:
        debug_steps['images'].extend([gray, canny_edges, dilated_edges])
        debug_steps['titles'].extend(['2. Grayscale', '3. Canny', '4. Dilated'])

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped_image = cv2.warpPerspective(image, M, (width, height))

    if debug:
        debug_steps['images'].append(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
        debug_steps['titles'].append('5. Final Result')
        debug_image = _create_debug_plot(debug_steps['images'], debug_steps['titles'], 'Normalization Steps', (2, 4))
        return warped_image, debug_image

    return warped_image, None

def segment_characters(image: np.ndarray, debug: bool = False) -> Tuple[List[Tuple[np.ndarray, Any]], Union[np.ndarray, None]]:
    """
    Phân đoạn các ký tự từ ảnh biển số đã được làm phẳng và sắp xếp chúng.
    """
    # --- CẤU HÌNH ---
    CLAHE_CLIP_LIMIT = 2.0
    ADAPTIVE_BLOCK_SIZE = 19
    ADAPTIVE_C = 9
    MIN_CHAR_HEIGHT_RATIO, MAX_CHAR_HEIGHT_RATIO = 0.25, 0.45
    MIN_ASPECT_RATIO, MAX_ASPECT_RATIO = 0.2, 0.8
    MIN_AREA = 100
    CHAR_RESIZE_DIM = (28, 28)
    
    debug_steps = {'images': [], 'titles': []}
    if debug:
        debug_steps['images'].append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        debug_steps['titles'].append('1. Normalized')

    # --- 1. Tiền xử lý ảnh ---
    v_channel = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
    enhanced = clahe.apply(v_channel)
    binary_img = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)
    
    if debug:
        debug_steps['images'].extend([v_channel, enhanced, binary_img])
        debug_steps['titles'].extend(['2. V-Channel', '3. CLAHE', '4. Binary'])
        
    # --- 2. Tìm và lọc Contours ---
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_candidates = []
    plate_h = image.shape[0]
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h) if h > 0 else 0
        height_ratio = h / float(plate_h) if plate_h > 0 else 0
        
        if (MIN_CHAR_HEIGHT_RATIO < height_ratio < MAX_CHAR_HEIGHT_RATIO and
            MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO and
            cv2.contourArea(c) > MIN_AREA):
            char_candidates.append(c)

    # --- 3. Sắp xếp ký tự theo 2 hàng ---
    final_chars = []
    if len(char_candidates) > 1:
        # Sắp xếp sơ bộ từ trên xuống dưới
        bounding_boxes = [cv2.boundingRect(c) for c in char_candidates]
        contours_boxes = sorted(zip(char_candidates, bounding_boxes), key=lambda b: b[1][1])
        
        # Tìm khoảng trống Y lớn nhất để xác định điểm chia hàng
        max_y_gap, split_index = 0, 4
        for i in range(len(contours_boxes) - 1):
            bottom_of_current = contours_boxes[i][1][1] + contours_boxes[i][1][3]
            top_of_next = contours_boxes[i+1][1][1]
            y_gap = top_of_next - bottom_of_current
            if y_gap > max_y_gap:
                max_y_gap = y_gap
                split_index = i + 1

        # Chia 2 hàng, sắp xếp mỗi hàng từ trái qua phải, rồi ghép lại
        top_row = sorted(contours_boxes[:split_index], key=lambda b: b[1][0])
        bottom_row = sorted(contours_boxes[split_index:], key=lambda b: b[1][0])
        sorted_contours = [item[0] for item in top_row + bottom_row]
        
        # Trích xuất ảnh ký tự
        for c in sorted_contours:
            (x, y, w, h) = cv2.boundingRect(c)
            char_crop = binary_img[y:y+h, x:x+w]
            char_resized = cv2.resize(char_crop, CHAR_RESIZE_DIM, interpolation=cv2.INTER_AREA)
            final_chars.append((char_resized, (x, y, w, h)))

    if debug:
        final_contours_img = image.copy()
        for _, (x, y, w, h) in final_chars:
            cv2.rectangle(final_contours_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        debug_steps['images'].append(cv2.cvtColor(final_contours_img, cv2.COLOR_BGR2RGB))
        debug_steps['titles'].append('5. Final Chars')
        debug_image = _create_debug_plot(debug_steps['images'], debug_steps['titles'], 'Segmentation Steps', (2, 3))
        return final_chars, debug_image
        
    return final_chars, None