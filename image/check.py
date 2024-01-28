import cv2
import numpy as np
# Đọc ảnh
original_image = cv2.imread('beard.png')
background_color = (0, 255, 0)
# Xác định tọa độ của điểm (ví dụ: điểm ở hàng 100, cột 150)
x1, y1 = 640, 270
# In tọa độ
# Hiển thị ảnh với điểm được đánh dấu
cv2.circle(original_image, (x1, y1), 5, (255, 255, 255), -1)

# Define the background color (BGR format)
background_color = (0, 255, 0)  # Green in this example

# Create a new image with the same size as the original and fill it with the background color
new_image = np.full_like(original_image, background_color, dtype=np.uint8)

# Overlay the original image on top of the new image
result = cv2.addWeighted(original_image, 1, new_image, 1, 0)

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
