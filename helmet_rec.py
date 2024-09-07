import cv2

a = cv2.CascadeClassifier("haarcascade_helmet.xml")
z = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
b = cv2.VideoCapture(0)

while True:
    c_dec, d_img = b.read()
    e = cv2.cvtColor(d_img, cv2.COLOR_BGR2GRAY)
    f = a.detectMultiScale(e, 1.3, 6)
    g = z.detectMultiScale(e, 1.3, 6)

    if len(f) > 0:
      
        for (x1, y1, w1, h1) in f:
            cv2.rectangle(d_img, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 5)
            cv2.putText(d_img, 'Helmet detected', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(d_img, 'Bike can start', (x1, y1 +h1+ 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
    else:
       
        for (x1, y1, w1, h1) in g:
            cv2.rectangle(d_img, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 5)
            cv2.putText(d_img, 'Helmet not detected', (x1 - 50, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
       
    cv2.imshow('img', d_img)
    h = cv2.waitKey(40) & 0xff
    if h == 27:
        break
b.release()
cv2.destroyAllWindows()
