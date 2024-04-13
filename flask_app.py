
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request
import face_recognition
import pickle
import cv2
import urllib.request
import os
import numpy as np
import imutils
import threading

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello from change!'

@app.route('/api/train_model', methods=['POST'])
def train_model_api():
    if request.method == 'POST':
        # Lấy dữ liệu từ request
        data = request.get_json()

        # Lấy danh sách các URL từ dữ liệu nhận được
        urls = data.get('urls', [])

        # Lấy userId từ dữ liệu nhận được
        userId = data.get('userId', '')
        train_model_from_list_urls_async(urls, userId)
        return "Training started"
    return 'failed'

def train_model_from_list_urls_async(urls, userId):
    threading.Thread(target=train_model_from_list_urls, args=(urls, userId)).start()


def read_image_from_url(url):
    # Đọc dữ liệu hình ảnh từ URL
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

# def read_image_from_url(url, max_size=2000, target_quality=70):
#     # Đọc dữ liệu hình ảnh từ URL
#     resp = urllib.request.urlopen(url)
#     image = np.asarray(bytearray(resp.read()), dtype="uint8")
#     image = cv2.imdecode(image, cv2.IMREAD_COLOR)

#     # Kiểm tra kích thước của hình ảnh
#     if image.shape[0] > max_size or image.shape[1] > max_size:
#         # Tính tỉ lệ giảm chất lượng để đạt được kích thước tối đa
#         ratio = max(image.shape[0], image.shape[1]) / max_size
#         target_quality = int(target_quality / ratio)

#         # Giảm chất lượng hình ảnh
#         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), target_quality]
#         _, img_encoded = cv2.imencode('.jpg', image, encode_param)
#         image = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)

#     return image

def train_model_from_list_urls(urls, userId):
    desired_width = 1000
    desired_height = 1000

    knownEncodings = []
    knownNames = []
    # loop over the image urls
    for url in urls:
        # load the image from url
        image = read_image_from_url(url)
        image = cv2.resize(image, (desired_width, desired_height))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Use Face_recognition to locate faces
        boxes = face_recognition.face_locations(rgb,model='hog')
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        # loop over the encodings
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(userId)
    #save encodings along with their names in dictionary data
    data = {"encodings": knownEncodings, "names": knownNames}
    #use pickle to save data into a file for later use
    path_save_model = f"model/{userId}.model"
    with open(path_save_model, "wb") as f:
        f.write(pickle.dumps(data))
    return data

def detect_face_from_url(url):
    image = read_image_from_url(url)
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(30, 30))

    # Initialize a list to store cropped faces
    cropped_faces = []

    # Crop faces and store them in the list
    for (x, y, w, h) in faces:
        cropped_faces.append(image[y:y+h, x:x+w])

    return cropped_faces

def predict_face(model, image):
    # Đường dẫn đến file xml chứa haarcascade
    cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    # Load haarcascade vào cascade classifier
    faceCascade = cv2.CascadeClassifier(cascPathface)

    # Độ phân giải mong muốn cho ảnh (ví dụ: 100x100)
    desired_width = 1000
    desired_height = 1000

    # Resize ảnh
    image = imutils.resize(image, width=desired_width, height=desired_height)

    # Chuyển đổi ảnh sang không gian màu RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Chuyển đổi ảnh sang ảnh xám để sử dụng Haarcascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Phát hiện các khuôn mặt trong ảnh
    faces = faceCascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=1,
                                        minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE)

    # Mã nhúng khuôn mặt trong ảnh đầu vào
    encodings = face_recognition.face_encodings(rgb)

    # Duyệt qua các mã nhúng để so sánh với dữ liệu đã biết
    for encoding in encodings:
        matches = face_recognition.compare_faces(model["encodings"], encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = model["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)

        if (name != "Unknown"):
            return True
    return False

def load_model(user_id):
    # Đường dẫn tới model
    path_model = f"model/{user_id}.model"

    # Đọc dữ liệu từ file model
    with open(path_model, "rb") as f:
        model_data = pickle.load(f)

    return model_data

# train_model_from_list_urls, args=(["https://i.ibb.co/sPzR3n6/2.jpg"], userId)

# print("load model")
# model = load_model(102200077)
# print("loaded model\ncrop face")
# cropped_faces = detect_face_from_url("https://i.ibb.co/mGNbBZF/5.jpg")
# print("croped")
# for face in cropped_faces:
#     print("predict:")
#     print(predict_face(model, face))
print("ok")




