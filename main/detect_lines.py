import cv2

file_name = '../data/test.jpg'
 
img  = cv2.imread(file_name)

img_final = cv2.imread(file_name)    
img_final = cv2.resize(img_final, (img_final.shape[1]*5, img_final.shape[0]*5))    
img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  
ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)    
image_final = cv2.bitwise_and(img2gray , img2gray , mask = mask)

ret, new_img = cv2.threshold(image_final, 180 , 255, cv2.THRESH_BINARY_INV)
    
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2 , 2))
dilated = cv2.dilate(new_img,kernel,iterations = 5)
    
_, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
    
index = 0     
for contour in contours: 
    [x,y,w,h] = cv2.boundingRect(contour)
  
    if w <1 and h<1:    
        continue

    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
    
    cropped = image_final[y :y +  h , x : x + w]
    
    s = '../output/crop_' + str(index) + '.jpg'     
    cv2.imwrite(s , cropped)    
    index = index + 1

cv2.imshow('captcha_result' , img)    
cv2.waitKey()
