import cv2

camera = 0
video = cv2.VideoCapture(camera)
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
id = int(input('Masukan id : '))
a = 0

while True:
    check, frame = video.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = faceDeteksi.detectMultiScale(abu, 1.3, 5)

    for (x, y, w, h) in wajah:
        cv2.imwrite('DataSet/User.' + str(id) + '.' + str(a) + '.jpg', abu[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if a >= 3:
        break

    a += 1

video.release()
cv2.destroyAllWindows()