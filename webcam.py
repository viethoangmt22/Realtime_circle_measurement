import cv2
import numpy as np
from ultilities import get_contours, warp_image

# Kích thước tờ giấy A4 (milimet)
w_a4 = 210
h_a4 = 297
scale = 1

# Khởi tạo webcam
cap = cv2.VideoCapture(0)
# codec = 0x47504A4D  # MJPG
cap.set(cv2.CAP_PROP_FPS, 30.0)
# cap.set(cv2.CAP_PROP_FOURCC, codec)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    contours = get_contours(frame, min_area=50000, draw_cont=False, show_canny=False)

    if len(contours) > 0:
        biggest_rect = contours[0][1]
        warp_img = warp_image(frame, biggest_rect, width=scale * w_a4, height=scale * h_a4)

        # Tỉ lệ để chuyển đổi pixel thành milimet (điều này phụ thuộc vào độ phân giải hình ảnh)
        pixel_to_mm = w_a4 / warp_img.shape[1]

        # Chuyển đổi ảnh sang ảnh grayscale
        gray_image = cv2.cvtColor(warp_img, cv2.COLOR_BGR2GRAY)

        # Sử dụng Gaussian Blur để làm mờ ảnh và làm nổi bật đường viền
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Sử dụng Hough Circle Transform để tìm các đường tròn trong ảnh
        circles = cv2.HoughCircles(
            blurred_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(100 / pixel_to_mm),  # Khoảng cách tối thiểu giữa các đường tròn
            param1=50,
            param2=30,
            minRadius=int(50 / pixel_to_mm),  # Bán kính tối thiểu
            maxRadius=int(150 / pixel_to_mm),  # Bán kính tối đa
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                diameter = 2 * radius

                # Vẽ đường viền hình tròn
                cv2.circle(warp_img, center, radius, (0, 255, 0), 2)

                # Hiển thị đường kính
                cv2.putText(
                    warp_img,
                    f"Diameter: {diameter:.2f} mm",
                    (circle[0] - 50, circle[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

        # Hiển thị hình ảnh đã xử lý
        
        cv2.imshow("Detected Circles", warp_img)
    cv2.imshow("test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
