import argparse
import time
import cv2


image = cv2.imread('../data/test2.jpg')
(winW, winH) = (32, 32)

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    
def pyramid(image, scale=1.5, minSize=(32, 32)):
    yield image

    while True:
        w = int(image.shape[1] / scale)
        image = resize(image, width=w)

        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

index = 0

for resized in pyramid(image, scale=1.5):
    for (x, y, window) in sliding_window(resized, stepSize=1, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        
        index = index + 1

        clone = resized.copy()
        clone2 = resized.copy()
        cropped = clone[y :y + winW , x : x + winW]
        
#         cv2.imwrite('../output2/cropped_' + str(index) + '.jpg', cropped)

        cv2.imshow('Window', cropped)
        cv2.waitKey(1)
        time.sleep(0.05)
