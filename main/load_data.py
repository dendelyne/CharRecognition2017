import struct
import numpy as np
import cv2
from PIL import Image, ImageEnhance



def read_record_ETL8G(f):
    s = f.read(8199)
    r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
    iF = Image.frombytes('F', (128, 127), r[14], 'bit', 4)
    iL = iF.convert('L')
    return r + (iL,)

def remove_duplicates(values):
    output = []
    seen = set()
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output

saveFile = []
imageList = []
numberOfTypes = 0
output = np.zeros([72, 160, 127, 128], dtype=np.uint8)

start_time = time.time()

for fileNumber in range(1, 33):
        filename = '../ETL8G/ETL8G_{:02d}'.format(fileNumber)
        with open(filename, 'rb') as f:
            for datasetNumber in range(5):
                index = 0
                for characterNumber in range(956):
                    r = read_record_ETL8G(f)
                    if b'.HIRA' in r[2]:
                        iE = Image.eval(r[-1], lambda x: 255-x*16)
                        im = np.array(iE)
                        blur = cv2.GaussianBlur(im,(5,5),0)
                        ret, thresh = cv2.threshold(blur,254,255,cv2.THRESH_BINARY)
                        output[index, (fileNumber - 1) * 5 + datasetNumber] = thresh
                        imageList.append(r[1])
                        numberOfTypes = len(remove_duplicates(imageList))
                        index = index + 1
                        
np.savez_compressed("../data/characters.npz", output=output, imageList=imageList, numberOfTypes=numberOfTypes)
print("done loading data!") 
