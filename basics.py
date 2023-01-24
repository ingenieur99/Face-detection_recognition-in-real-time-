import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('imagesBasic/Elon Musk.jpg')
#conversion de l'image
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

imgtest = face_recognition.load_image_file('imagesBasic/elon test.jpg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

#second step : finding faces in imagesand then finding there encodings as well(cadre du visage)
faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLoctest = face_recognition.face_locations(imgtest)[0]
encodeElontest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#third step comparing the faces and finding the distance between them


results = face_recognition.compare_faces([encodeElon], encodeElontest)
faceDis = face_recognition.face_distance([encodeElon], encodeElontest)
print(results, faceDis)
cv2.putText(imgtest , f'{results}{round(faceDis[0],2)}' , (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)




cv2.imshow('elon musk', imgElon)
cv2.imshow('elon test', imgtest)
cv2.waitKey(0)
