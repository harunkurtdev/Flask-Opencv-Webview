import os
from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
# Flask başlatma
app = Flask(__name__)

#dosya yükleme işlemleri için izin alıyoruz =) (öyle diyelim)
UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#ilk açılış sayfas
@app.route("/")
def start_page():
    print("Start")
    return render_template('index.html')

# resim işleme sayfası
@app.route('/upload', methods=['POST'])
def upload_file():
    #gelen request ile gelen resmi alıyoruz
    file = request.files['image']

    # gelen Resimi kaydetme
    #filename = 'static/' + file.filename
    #file.save(filename)

    # Resim okuma
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    
    # Fonksiyon ile bulma
    faces = detect_faces(image)
    #eğer faces sıfır ise resim de nesne yoktur
    if len(faces) == 0:
        faceDetected = False
        num_faces = 0
        to_send = ''
    else:
        faceDetected = True
        num_faces = len(faces)
        
        # yüzlere kare çizdirmel istiyor isek
        for item in faces:
            draw_rectangle(image, item['rect'])
        
        # resmi kaydetmek istiyprsak
        #cv2.imwrite(filename, image)
        
        # base64 e çevirerek resmi resmi html olarak gönderme
        image_content = cv2.imencode('.jpg', image)[1].tostring()
        encoded_image = base64.encodestring(image_content)
        to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')

    return render_template('index.html', faceDetected=faceDetected, num_faces=num_faces, image_to_show=to_send, init=True)

# ----------------------------------------------------------------------------------
# REsim de Yüz bulma OpenCV
# ----------------------------------------------------------------------------------  
def detect_faces(img):
    '''REsim de Yüz bulma '''
    
    faces_list = []

    # opencv bir şeyleri bulabilmesi için resimleri gri skalaya çeviririz
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Burada cascade imizi yüklüyoruz
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    #burada mutli ile birden fazla cascade işlemi yapabilmek için kullanıyoruz
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    # eğer resimde cascade en kimse bulunmamış ise geri çevir
    if  len(faces) == 0:
        return faces_list
    
    for i in range(0, len(faces)):
        (x, y, w, h) = faces[i]
        face_dict = {} # dizi ile maps veri tipinde olmasından bahsediyoruz
        face_dict['face'] = gray[y:y + w, x:x + h] # burada face ve
        face_dict['rect'] = faces[i] # rect adında bir yere atadık
        faces_list.append(face_dict) # face list e aktarıyoruz face değişkenlerini

    #
    return faces_list
# ----------------------------------------------------------------------------------
# Kare çizdirme yeri
# x ve y ye göre yani pt1 ve pt2 göre kare çiziyor
# ----------------------------------------------------------------------------------
def draw_rectangle(img, rect):

    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

#başlatma
if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='localhost', debug=True, port=9886)