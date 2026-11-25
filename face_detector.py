import cv2

cascPath = "src\\face.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
 
while True:
    # membaca video capture
    rect, frame = video_capture.read()    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:            
        center_coordinates = x + w // 2, y + h // 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  
        #radius = w // 2
        #cv2.circle(frame, center_coordinates, radius, (0, 0, 100), 3) #ini buat print lingkaran saat mendeteksi wajah
        #jadi pilih salah satu aja bisa kotak atau lingkaran



    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()