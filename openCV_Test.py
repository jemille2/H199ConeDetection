import numpy as np
import cv2

cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
	ret,frame = cap.read()
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.putText(gray, 'sample text', (350,130), font, 1, (200,255,155), 2, cv2.LINE_AA)
	
	cv2.imshow('window label', gray)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()