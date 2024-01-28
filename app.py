from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
import numpy as np
import cv2
from PIL import Image, ImageDraw
import io
import tensorflow as tf

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Đường dẫn đến model TensorFlow Lite
model_path = 'deeplab.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Các thông số của model (có thể cần điều chỉnh tùy thuộc vào model của bạn)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

global_keypoint = []

@app.route('/detect', methods=['POST'])
@cross_origin(origins='*')
def detect():
    try:
        global global_keypoint
        image_file = request.files['image']
        if image_file:
            print("Đã nhận được dữ liệu hình ảnh từ yêu cầu POST")
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (512, 512))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            img = img.astype(np.float32)
            # Đặt dữ liệu vào input tensor
            interpreter.set_tensor(input_details[0]['index'], img)

            # Thực hiện dự đoán
            interpreter.invoke()

            # Lấy kết quả từ output tensor
            predictions = interpreter.get_tensor(output_details[0]['index'])

            keypoints = []
            for i in range(predictions.shape[-1]):
                coord = np.unravel_index(predictions[0, :, :, i].argmax(), predictions[0, :, :, i].shape)[::-1]
                keypoints.append(coord)

            global_keypoint = keypoints
            return jsonify({'ok': 'detected'})
        else:
            return jsonify({'error': 'Không có file hình ảnh được gửi'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/glasses', methods=['POST'])
@cross_origin(origins='*')
def glasses():
    try:
        # Lấy dữ liệu hình ảnh từ yêu cầu POST
        image_file = request.files['image']
        # Đảm bảo rằng file hình ảnh được gửi điểm kèm theo
        if image_file:
            print("Đã nhận được dữ liệu hình ảnh từ yêu cầu POST")
            # Đọc và xử lý hình ảnh
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            # Resize ảnh về kích thước 512x512
            img = cv2.resize(img, (512, 512))
            img_original = img.copy()  # Sao chép hình ảnh gốc

            # Vẽ glasses dựa trên toạ độ keypoints 0 và 1
            img_with_glasses = draw_glasses(img_original, global_keypoint[0], global_keypoint[1])      
            # # Lưu ảnh đã được đánh dấu tạm thời trong bộ nhớ
            image_io = io.BytesIO()
            Image.fromarray(cv2.cvtColor(img_with_glasses, cv2.COLOR_BGR2RGB)).save(image_io, format='JPEG')
            image_io.seek(0)

            # Trả về ảnh đã được đánh dấu
            return send_file(image_io, mimetype='image/jpeg')

        else:
            return jsonify({'error': 'Không có file hình ảnh được gửi'})

    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/hat', methods=['POST'])
@cross_origin(origins='*')
def hat():
    try:
        # Lấy dữ liệu hình ảnh từ yêu cầu POST
        image_file = request.files['image']

        # Đảm bảo rằng file hình ảnh được gửi điểm kèm theo
        if image_file:
            print("Đã nhận được dữ liệu hình ảnh từ yêu cầu POST")

            # Đọc và xử lý hình ảnh
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

            # Resize ảnh về kích thước 512x512
            img = cv2.resize(img, (512, 512))

            img_original = img.copy()  # Sao chép hình ảnh gốc
            # Vẽ glasses dựa trên toạ độ keypoints 0 và 1
            img_with_glasses = draw_hat(img_original, global_keypoint[3], global_keypoint[8])      
            # Lưu ảnh đã được đánh dấu tạm thời trong bộ nhớ
            image_io = io.BytesIO()
            Image.fromarray(cv2.cvtColor(img_with_glasses, cv2.COLOR_BGR2RGB)).save(image_io, format='JPEG')
            image_io.seek(0)

            # Trả về ảnh đã được đánh dấu
            return send_file(image_io, mimetype='image/jpeg')

        else:
            return jsonify({'error': 'Không có file hình ảnh được gửi'})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/beard', methods=['POST'])
@cross_origin(origins='*')
def beard():
    try:
        # Lấy dữ liệu hình ảnh từ yêu cầu POST
        image_file = request.files['image']

        # Đảm bảo rằng file hình ảnh được gửi điểm kèm theo
        if image_file:
            print("Đã nhận được dữ liệu hình ảnh từ yêu cầu POST")

            # Đọc và xử lý hình ảnh
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

            # Resize ảnh về kích thước 512x512
            img = cv2.resize(img, (512, 512))

            img_original = img.copy()  # Sao chép hình ảnh gốc
            # Vẽ glasses dựa trên toạ độ keypoints 0 và 1
            img_with_glasses = draw_beard(img_original, global_keypoint[2], global_keypoint[0], global_keypoint[1])      
            # Lưu ảnh đã được đánh dấu tạm thời trong bộ nhớ
            image_io = io.BytesIO()
            Image.fromarray(cv2.cvtColor(img_with_glasses, cv2.COLOR_BGR2RGB)).save(image_io, format='JPEG')
            image_io.seek(0)

            # Trả về ảnh đã được đánh dấu
            return send_file(image_io, mimetype='image/jpeg')

        else:
            return jsonify({'error': 'Không có file hình ảnh được gửi'})

    except Exception as e:
        return jsonify({'error': str(e)})

# def draw_keypoints_with_coordinates(image, keypoints):
#     pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(pil_image)

#     # Vẽ chấm tròn màu đỏ tại mỗi keypoint và hiển thị tọa độ
#     for i, point in enumerate(keypoints):
#         draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill='red')
#         draw.text((point[0] + 10, point[1] - 5), f"{i}: {point}", fill='red')

#     return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def draw_glasses(image, point0, point1):
    try:
        with open('image/glasses.txt', 'r') as file:
            line = file.readline()
            x1, y1, x2, y2 = map(int, line.split())
    except Exception as e:
        print(f"Error reading glasses.txt: {e}")
        return image

    # Check if coordinates are finite numbers
    if not all(np.isfinite([x1, y1, x2, y2])):
        print("Invalid coordinates in glasses.txt")
        return image

    # Check if coordinates are within the valid range
    if not (0 <= x1 <= x2 and 0 <= y1 <= y2):
        print("Invalid coordinates range in glasses.txt")
        return image

    # Check if scale denominator is not zero
    if abs(y2 - y1) == 0:
        scale_y = 1
    else:
        scale_y = abs(point1[1] - point0[1]) / abs(y2 - y1)

    # Vẽ glasses dựa trên toạ độ keypoints 0 và 1
    glasses_image = cv2.imread('image/glasses.png', cv2.IMREAD_UNCHANGED)
    h, w, _ = glasses_image.shape

    # Chuyển đổi kích thước và vị trí của glasses
    scale_x = abs(point1[0] - point0[0]) / abs(x2 - x1)
    
    # Check if scale values are finite numbers
    if not all(np.isfinite([scale_x, scale_y])):
        print("Invalid scale values for glasses")
        return image

    glasses_resized = cv2.resize(glasses_image, (int(w * scale_x), int(h * scale_y)))

    # Tính toán vị trí điểm anchor trên ảnh gốc
    # Tính toán toạ độ mới của điểm anchor
    anchor_x = point0[0] - int(x1 * scale_x)
    anchor_y = point0[1] - int(y1 * scale_y)
    
    # Vẽ glasses lên ảnh
    overlay_image(image, glasses_resized, anchor_x, anchor_y)
    return image

def draw_hat(image, point3, point8):
    try:
        with open('image/hat.txt', 'r') as file:
            line = file.readline()
            x1, y1, x2, y2 = map(int, line.split())
    except Exception as e:
        print(f"Error reading hat.txt: {e}")
        return image

    # Check if coordinates are finite numbers
    if not all(np.isfinite([x1, y1, x2, y2])):
        print("Invalid coordinates in hat.txt")
        return image

    # Check if coordinates are within the valid range
    if not (0 <= x1 <= x2 and 0 <= y1 <= y2):
        print("Invalid coordinates range in hat.txt")
        return image

    # Check if scale denominator is not zero
    if abs(y2 - y1) == 0:
        scale_y = 1
    else:
        scale_y = abs(point8[1] - point3[1]) / abs(y2 - y1)

    # Vẽ hat dựa trên toạ độ keypoints 3 và 8
    hat_image = cv2.imread('image/hat.png', cv2.IMREAD_UNCHANGED)
    h, w, _ = hat_image.shape

    # Chuyển đổi kích thước và vị trí của hat
    scale_x = abs(point8[0] - point3[0]) / abs(x2 - x1)
    
    # Check if scale values are finite numbers
    if not all(np.isfinite([scale_x, scale_y])):
        print("Invalid scale values for hat")
        return image

    hat_resized = cv2.resize(hat_image, (int(w * scale_x), int(h * scale_y)))

    # Tính toán vị trí điểm anchor trên ảnh gốc
    # Tính toán toạ độ mới của điểm anchor
    anchor_x = point3[0] - int(x1 * scale_x)
    anchor_y = point3[1] - int(y1 * scale_y)

    # Kiểm tra xem có cần crop hình hat không
    hat_crop = crop_image(hat_resized, anchor_x, anchor_y, image.shape[1], image.shape[0])

    # Vẽ hat lên ảnh
    overlay_image(image, hat_crop, anchor_x, anchor_y)
    return image

def draw_beard(image, point, point1, point2):
    try:
        with open('image/beard.txt', 'r') as file:
            line = file.readline()
            x1, y1 = map(int, line.split())
    except Exception as e:
        print(f"Error reading beard.txt: {e}")
        return image

    # Check if coordinates are finite numbers
    if not all(np.isfinite([x1, y1])):
        print("Invalid coordinates in beard.txt")
        return image

    # Vẽ beard dựa trên toạ độ từ file beard.txt
    beard_image = cv2.imread('image/beard.png', cv2.IMREAD_UNCHANGED)
    h, w, _ = beard_image.shape

    # Tính toán kích thước mới dựa trên khoảng cách giữa point1 và point2
    new_width = int(np.linalg.norm(np.array(point2) - np.array(point1)))

    # Resize beard_image
    scale_factor = new_width / w
    beard_resized = cv2.resize(beard_image, (int(w * scale_factor), int(h * scale_factor)))

    # Tính toán vị trí điểm anchor trên ảnh gốc
    # anchor_x = int((point1[0] + point2[0]) / 2) - int(x1 * scale_factor)
    # anchor_y = int((point1[1] + point2[1]) / 2) - int(y1 * scale_factor)
    anchor_x = point[0] - int(x1 * scale_factor)
    anchor_y = point[1] - int(y1 * scale_factor)
    # Kiểm tra xem có cần crop hình beard không
    beard_crop = crop_image(beard_resized, anchor_x, anchor_y, image.shape[1], image.shape[0])

    # Vẽ beard lên ảnh
    overlay_image(image, beard_crop, anchor_x, anchor_y)
    return image

def crop_image(image, x, y, width, height):
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + image.shape[1], width)
    y2 = min(y + image.shape[0], height)

    return image[y1 - y:y2 - y, x1 - x:x2 - x, :]

def overlay_image(background, overlay, x, y):
    h, w, _ = overlay.shape

    # Tính toán vị trí để chèn overlay lên background
    y1, y2 = y, y + h
    x1, x2 = x, x + w

    # Xác định phần ảnh overlay cần chèn
    overlay_image = overlay[:, :, :3]

    # Xác định phần alpha channel của overlay
    alpha_mask = overlay[:, :, 3] / 255.0

    # Điều chỉnh toạ độ để vẽ overlay đúng vị trí trên background
    x_offset = max(0, x1)  # Đảm bảo x_offset không nhỏ hơn 0
    y_offset = max(0, y1)  # Đảm bảo y_offset không nhỏ hơn 0

    overlay_width = min(w, background.shape[1] - x_offset)
    overlay_height = min(h, background.shape[0] - y_offset)

    x1_bg, x2_bg = x_offset, x_offset + overlay_width
    y1_bg, y2_bg = y_offset, y_offset + overlay_height


    # Chèn ảnh overlay lên background
    for c in range(0, 3):
        background[y1_bg:y2_bg, x1_bg:x2_bg, c] = (1 - alpha_mask) * background[y1_bg:y2_bg, x1_bg:x2_bg, c] + alpha_mask * overlay_image[:overlay_height, :overlay_width, c]
if __name__ == '__main__':
    app.run()
    #app.run(host='0.0.0.0', port=8888, ssl_context=("cert.pem", "key.pem"))
