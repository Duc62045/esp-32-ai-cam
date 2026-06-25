import cv2
import numpy as np
import urllib.request
import os
from object_groups import object_groups  # Import từ file riêng
import webcolors
from css3_colors import css3_colors
import mediapipe as mp
import math

mp_pose = mp.solutions.pose

confThreshold = 0.6
nmsThreshold = 0.4
# **Đường dẫn đến các file mô hình YOLO và Caffe**
yolo_weights = "Module\yolov3.weights"
yolo_cfg = "Module\yolov3.cfg"
coco_names = "Module\coco.names"
model_def = "Module\deploy-vgg16.prototxt"
model_weights = "Module\minc-vgg16.caffemodel"
categories_file = "Module\categories.txt"

# **Kiểm tra sự tồn tại của các file**
for file in [yolo_weights, yolo_cfg, coco_names, model_def, model_weights, categories_file]:
    if not os.path.exists(file):
        print(f"File không tồn tại: {file}")
        exit()

# **Tải danh sách lớp từ COCO và MINC**
with open(coco_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]
with open(categories_file, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

# **Tải mô hình YOLO và Caffe**
yolo_net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
caffe_net = cv2.dnn.readNetFromCaffe(model_def, model_weights)

# **Hàm xử lý thuộc tính**
def get_object_group_and_attributes(label):
    for group, data in object_groups.items():
        if label in data["objects"]:
            return group, data["attributes"]
    return "unknown", []

def closest_color(requested_color):
    """Tìm tên màu gần nhất dựa trên mã RGB."""
    min_distance = float("inf")
    closest_name = None

    for name, hex_value in css3_colors.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        distance = rd + gd + bd

        if distance < min_distance:
            min_distance = distance
            closest_name = name

    return closest_name


def get_dominant_color(image, x, y, w, h):
    """Trích xuất màu sắc chủ đạo từ vùng bounding box."""
    object_img = image[max(0, y):y+h, max(0, x):x+w]
    object_img = cv2.cvtColor(object_img, cv2.COLOR_BGR2RGB)
    pixels = object_img.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    dominant_color = unique_colors[np.argmax(counts)]

    try:
        # Tìm tên màu chính xác
        return webcolors.rgb_to_name(tuple(dominant_color))
    except ValueError:
        # Nếu không tìm thấy tên màu chính xác, trả về màu gần nhất
        return closest_color(tuple(dominant_color))


def get_material(image, x, y, w, h):
    object_img = image[max(0, y):y+h, max(0, x):x+w]
    object_img = cv2.resize(object_img, (224, 224))
    object_img = object_img.astype(np.float32) - np.array([104, 117, 123])
    object_img = object_img.transpose((2, 0, 1))
    caffe_net.setInput(np.expand_dims(object_img, axis=0))
    output = caffe_net.forward()
    material_id = np.argmax(output)
    return labels[material_id]

def get_state(image, x, y, w, h, label):
    object_img = image[max(0, y):y+h, max(0, x):x+w]
    if label == "bottle":
        dominant_color = get_dominant_color(object_img, 0, 0, w, h)
        if "white" in dominant_color.lower():  # Giả định cho trạng thái
            return "Empty"
        return "Full"
    return "Unknown"

def get_condition(image, x, y, w, h, label):
    object_img = image[max(0, y):y+h, max(0, x):x+w]
    gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    num_edges = np.sum(edges > 0)
    if num_edges > 1000:
        return "Worn"
    return "New"

def get_details(image, x, y, w, h):
    # Loại bỏ hoặc không sử dụng hàm này.
    return "Details not analyzed (OCR skipped for performance)."

def get_brand(image, x, y, w, h):
    # Không thực hiện OCR, trả về giá trị mặc định
    return "Brand not analyzed"


def get_ports(image, x, y, w, h):
    return "HDMI, USB"

def get_features(image, x, y, w, h, label):
    if label == "car":
        return "Headlights, wheels"
    elif label == "bicycle":
        return "Wheels, pedals"
    return "Unknown features"

def get_relative_size(w, h, image_width, image_height):
    relative_area = (w * h) / (image_width * image_height)
    if relative_area > 0.1:
        return "Large"
    elif relative_area > 0.01:
        return "Medium"
    else:
        return "Small"

def get_shape(image, x, y, w, h):
    object_img = image[max(0, y):y+h, max(0, x):x+w]
    gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 3:
            return "Triangle"
        elif len(approx) == 4:
            return "Rectangle"
        elif len(approx) > 4:
            return "Circle/Ellipse"
    return "Unknown"

def get_packaging(image, x, y, w, h):
    object_img = image[max(0, y):y+h, max(0, x):x+w]

    gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    white_ratio = np.sum(binary == 255) / (binary.shape[0] * binary.shape[1])

    if white_ratio > 0.7:
        return "Plastic"  
    elif white_ratio > 0.4:
        return "Paper"  
    else:
        return "Box" 

def calculate_angle(point1, point2, point3):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3

    vector1 = (x1 - x2, y1 - y2)
    vector2 = (x3 - x2, y3 - y2)

    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

    if magnitude1 * magnitude2 == 0:  # Tránh chia cho 0
        return 0
    angle = math.acos(dot_product / (magnitude1 * magnitude2))
    return math.degrees(angle)

def get_pose(image, x, y, w, h):
    person_img = image[y:y + h, x:x + w]
    if person_img.size == 0:
        return "Empty Image"

    try:
        person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    except cv2.error:
        return "Error Converting to RGB"

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(person_img_rgb)
        if results.pose_landmarks:
            return "Pose Detected"
        else:
            return "Pose Not Detected"


def get_clothing(image, x, y, w, h):
    """Phân tích màu quần áo của người dựa trên các vùng áo và quần"""
    # Kiểm tra bounding box hợp lệ
    if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        return "Invalid Bounding Box"

    # Cắt vùng ảnh người
    person_img = image[y:y + h, x:x + w]

    # Kiểm tra vùng ảnh rỗng
    if person_img.size == 0:
        return "Empty Image"

    try:
        # Chuyển sang không gian màu HSV
        person_img_hsv = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
    except cv2.error:
        return "Error Converting to HSV"

    # Chia ảnh thành vùng áo (phần trên) và quần (phần dưới)
    upper_part = person_img_hsv[:h // 2, :]
    lower_part = person_img_hsv[h // 2:, :]

    # Ngưỡng màu cho các màu phổ biến
    colors = {
        "Blue": ([100, 150, 0], [140, 255, 255]),
        "Red (low)": ([0, 50, 50], [10, 255, 255]),  # Vùng đỏ thấp
        "Red (high)": ([170, 50, 50], [180, 255, 255]),  # Vùng đỏ cao
        "Green": ([40, 50, 50], [80, 255, 255]),
        "Yellow": ([20, 100, 100], [30, 255, 255]),
        "Orange": ([10, 100, 100], [20, 255, 255]),
        "Purple": ([130, 50, 50], [160, 255, 255]),
        "Pink": ([140, 50, 50], [170, 255, 255]),
        "White": ([0, 0, 200], [180, 50, 255]),
        "Black": ([0, 0, 0], [180, 255, 50]),
        "Gray": ([0, 0, 50], [180, 50, 200]),
    }

    # Hàm phụ để tìm màu chiếm ưu thế nhất trong vùng
    def dominant_color(region_hsv):
        max_color = None
        max_area = 0

        for color_name, (lower, upper) in colors.items():
            mask = cv2.inRange(region_hsv, np.array(lower), np.array(upper))
            color_area = np.sum(mask > 0)

            # Cập nhật màu chiếm ưu thế nhất
            if color_area > max_area:
                max_area = color_area
                max_color = color_name.split(" ")[0]  # Loại bỏ thông tin "low"/"high" ở màu đỏ

        return max_color if max_area > 500 else "Unknown"

    # Phân tích vùng áo và quần
    upper_color = dominant_color(upper_part)
    lower_color = dominant_color(lower_part)

    # Trả về kết quả
    return f"Wearing {upper_color} Shirt and {lower_color} Pants"



def get_action(image, x, y, w, h):
    """Xác định hành động của người dựa trên trạng thái và hình dạng"""
    # Cắt vùng ảnh người
    person_img = image[y:y + h, x:x + w]

    # Kiểm tra vùng ảnh rỗng
    if person_img.size == 0:
        return "Empty Image"

    # Chuyển ảnh sang grayscale
    person_img_gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)

    # Phát hiện cạnh bằng Canny
    edges = cv2.Canny(person_img_gray, 50, 150)

    # Đếm số điểm cạnh
    num_edges = np.sum(edges > 0)

    # Phát hiện các contours (đường bao)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tính diện tích lớn nhất từ các contours
    max_area = max(cv2.contourArea(cnt) for cnt in contours) if contours else 0

    # Dựa vào số cạnh và diện tích để xác định hành động
    if num_edges > 2000 and max_area > 5000:
        return "Running or Moving"
    elif num_edges < 500 and max_area < 3000:
        return "Standing Still"
    elif max_area > 3000 and num_edges < 1500:
        return "Sitting"
    else:
        return "Unknown Action"


# **Tải ảnh đầu vào**
import urllib.request

import time

def fetch_image_from_esp32(url, retry_limit=5):
    for attempt in range(retry_limit):
        try:
            img_resp = urllib.request.urlopen(url, timeout=5)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            image = cv2.imdecode(imgnp, -1)
            return image
        except Exception as e:
            print(f"Thử lần {attempt + 1}/{retry_limit}: Lỗi khi tải ảnh từ ESP32-CAM: {e}")
            time.sleep(2)
    print("Không thể tải ảnh từ ESP32-CAM sau nhiều lần thử.")
    return None

def findObject(outputs, image):
    hT, wT, cT = image.shape
    bbox = []
    classIds = []
    confs = []
    analyzed_objects = []  # Danh sách chứa thông tin đối tượng và thuộc tính đã phân tích

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = bbox[i]
            label = classes[classIds[i]]
            confidence = confs[i]

            # Lấy nhóm và thuộc tính của đối tượng
            group, attributes = get_object_group_and_attributes(label)

            # Phân tích các thuộc tính
            analyzed_attributes = {}
            if "color" in attributes:
                analyzed_attributes["color"] = get_dominant_color(image, x, y, w, h)
            if "material" in attributes:
                analyzed_attributes["material"] = get_material(image, x, y, w, h)
            if "size" in attributes:
                analyzed_attributes["size"] = get_relative_size(w, h, wT, hT)
            if "shape" in attributes:
                analyzed_attributes["shape"] = get_shape(image, x, y, w, h)
            if "state" in attributes:
                analyzed_attributes["state"] = get_state(image, x, y, w, h, label)
            if "details" in attributes:
                analyzed_attributes["details"] = get_details(image, x, y, w, h)
            if "condition" in attributes:
                analyzed_attributes["condition"] = get_condition(image, x, y, w, h, label)
            if "type" in attributes:
                analyzed_attributes["type"] = label
            if "brand" in attributes:
                analyzed_attributes["brand"] = get_brand(image, x, y, w, h)
            if "ports" in attributes:
                analyzed_attributes["ports"] = get_ports(image, x, y, w, h)
            if "features" in attributes:
                analyzed_attributes["features"] = get_features(image, x, y, w, h, label)
            if "packaging" in attributes:
                analyzed_attributes["packaging"] = get_packaging(image, x, y, w, h)
            if "pose" in attributes:
                analyzed_attributes["pose"] = get_pose(image, x, y, w, h)
            if "clothing" in attributes:
                analyzed_attributes["clothing"] = get_clothing(image, x, y, w, h)
            if "action" in attributes:
                analyzed_attributes["action"] = get_action(image, x, y, w, h)

            # Thêm vào danh sách phân tích
            analyzed_objects.append({
                "bbox": (x, y, w, h),
                "label": label,
                "confidence": confidence,
                "attributes": analyzed_attributes
            })

    return analyzed_objects
esp32_url = "http://192.168.51.26/cam-lo.jpg"  
while True:
    image = fetch_image_from_esp32(esp32_url)
    if image is None:
        print("Không thể tải ảnh từ ESP32-CAM. Đang thử lại...")
        continue 

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(output_layers)

    detected_objects = findObject(outputs, image)

    for obj in detected_objects:
        x, y, w, h = obj["bbox"]
        label = obj["label"]
        confidence = obj["confidence"]
        attributes = obj["attributes"]

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{label} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        y_offset = y + h + 15
        for attr, value in attributes.items():
            text = f"{attr.capitalize()}: {value}"
            cv2.putText(image, text, (x, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 15

    cv2.imshow("Detection", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()



height, width, channels = image.shape
while True:
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)


class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Ngưỡng tin cậy
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# **Áp dụng Non-Maximum Suppression (NMS)**
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# **Duyệt qua các đối tượng phát hiện**
for i in range(len(boxes)):
    if i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        # **Xác định nhóm đồ vật**
        group, attributes = get_object_group_and_attributes(label)

        # **Phân tích thuộc tính**
        analyzed_attributes = {}
        if "color" in attributes:
            analyzed_attributes["color"] = get_dominant_color(image, x, y, w, h)
        if "material" in attributes:
            analyzed_attributes["material"] = get_material(image, x, y, w, h)
        if "size" in attributes:
            analyzed_attributes["size"] = get_relative_size(w, h, width, height)
        if "shape" in attributes:
            analyzed_attributes["shape"] = get_shape(image, x, y, w, h)
        if "state" in attributes:
            analyzed_attributes["state"] = get_state(image, x, y, w, h, label)
        if "details" in attributes:
            analyzed_attributes["details"] = get_details(image, x, y, w, h)
        if "condition" in attributes:
            analyzed_attributes["condition"] = get_condition(image, x, y, w, h, label)
        if "type" in attributes:
            analyzed_attributes["type"] = label
        if "brand" in attributes:
            analyzed_attributes["brand"] = get_brand(image, x, y, w, h)
        if "ports" in attributes:
            analyzed_attributes["ports"] = get_ports(image, x, y, w, h)
        if "features" in attributes:
            analyzed_attributes["features"] = get_features(image, x, y, w, h, label)
        if "packaging" in attributes:
            analyzed_attributes["packaging"] = get_packaging(image, x, y, w, h)
        if "pose" in attributes:
            analyzed_attributes["pose"] = get_pose(image, x, y, w, h)
        if "clothing" in attributes:
            analyzed_attributes["clothing"] = get_clothing(image, x, y, w, h)
        if "action" in attributes:
            analyzed_attributes["action"] = get_action(image, x, y, w, h)
        # **Hiển thị lên ảnh**
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Bounding box
        cv2.putText(image, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Hiển thị tất cả các thuộc tính bên trái bounding box
        x_offset = x - 10  # Đặt vị trí bên trái bounding box
        y_offset = y

        for attr, value in analyzed_attributes.items():
            text = f"{attr.capitalize()}: {value}"

            # Vẽ nền mờ cho văn bản
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_width, text_height = text_size[0], text_size[1]
            cv2.rectangle(image, (x_offset - text_width - 10, y_offset - text_height - 2), 
                        (x_offset, y_offset + 2), (0, 0, 0), -1)  # Nền đen mờ

            # Vẽ văn bản
            cv2.putText(image, text, (x_offset - text_width - 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)  # Chữ trắng
            y_offset += 20  # Tăng khoảng cách giữa các dòng

# **Lưu ảnh kết quả**




# Tiền xử lý với YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
yolo_net.setInput(blob)
outputs = yolo_net.forward(output_layers)

# Phát hiện đối tượng
detected_objects = findObject(outputs, image)

# Vẽ bounding box và hiển thị thuộc tính
for obj in detected_objects:
    x, y, w, h = obj["bbox"]
    label = obj["label"]
    confidence = obj["confidence"]
    attributes = obj["attributes"]

    # Hiển thị thuộc tính
    y_offset = y + h + 15  # Hiển thị bên dưới bounding box
    for attr, value in attributes.items():
        text = f"{attr.capitalize()}: {value}"
        cv2.putText(image, text, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 15

cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)  # Cho phép thay đổi kích thước
cv2.imshow("Detection", image)

# Đặt cửa sổ hiển thị bằng kích thước của ảnh
height, width, _ = image.shape
cv2.resizeWindow("Detection", width, height)

cv2.waitKey(0)
cv2.destroyAllWindows()

