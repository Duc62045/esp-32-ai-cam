import cv2

# Bước 1: Mở camera (0 là chỉ số của camera mặc định, có thể thay đổi nếu sử dụng nhiều camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

# Bước 2: Đọc video từ camera
while True:
    ret, frame = cap.read()  # Đọc một khung hình từ camera
    if not ret:
        print("Không thể đọc khung hình từ camera.")
        break

    # Bước 3: Hiển thị khung hình
    cv2.imshow('Camera Feed', frame)

    # Bước 4: Dừng khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bước 5: Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
