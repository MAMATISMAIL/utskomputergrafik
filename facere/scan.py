import cv2

camera = 0
video = cv2.VideoCapture(camera)
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
a = 0

while True:
    check, frame = video.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = faceDeteksi.detectMultiScale(abu, 1.3, 5)

    for (x, y, w, h) in wajah:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "MANUSIA", (x+40, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1)
    if key == ord('a'):
        break

video.release()
cv2.destroyAllWindows()