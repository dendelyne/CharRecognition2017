import struct
import numpy as np
import time
import cv2
from statsmodels.tools import categorical
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
# output = []
numberOfTypes = 0
output = np.zeros([72, 160, 127, 128], dtype=np.uint8)

start_time = time.time()

for fileNumber in range(1, 33):
        filename = '../ETL8G/ETL8G_{:02d}'.format(fileNumber)
        with open(filename, 'rb') as f:
            for datasetNumber in range(5):
                index = 0
#                 for characterNumber in range(956):
                for characterNumber in range(956):
                    r = read_record_ETL8G(f)
#                     print(r)
                    if b'.HIRA' in r[2]:
                        iE = Image.eval(r[-1], lambda x: 255-x*16)
                        im = np.array(iE)
                        blur = cv2.GaussianBlur(im,(5,5),0)
                        ret, thresh = cv2.threshold(blur,254,255,cv2.THRESH_BINARY)
                        output[index, (fileNumber - 1) * 5 + datasetNumber] = thresh
                        imageList.append(r[1])
                        numberOfTypes = len(remove_duplicates(imageList))
                        index = index + 1
#                         print(index, (fileNumber - 1) * 5 + datasetNumber)
#                         print(characterNumber)
#                         print(thresh)
#                         Image.fromarray(thresh).show()
                        
#                         iE = Image.eval(r[-1], lambda x: 255-x*16)
#                         iE = iE.resize((32, 32))
#                         im = np.array(iE)
#     #                         gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#                         blur = cv2.GaussianBlur(im,(5,5),0)
#                         thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
#     #                         print("!!!!!!")
#     #                         print(np.asarray(thresh))
#                         imageList.append(r[1])
#                         output.append(thresh)
#                         numberOfTypes = len(remove_duplicates(imageList))
#                         imageList2 = remove_duplicates(imageList)
# #                         print(saveFile)
#                         img = Image.fromarray(thresh)
# #                         img.save('my.png')
# # #                         img.show()
# #                         fn = '../output/ETL8G_{:04d}_{:04d}.png'.format(r[1], numberOfTypes)
# #                         img.save(fn, 'PNG')
                        
    
# np.savez_compressed("../data/characters.npz", ary=ary, imageList=imageList, imageList2=imageList2, output=output, numberOfTypes=numberOfTypes)
np.savez_compressed("../data/hiragana.npz", output=output, imageList=imageList, numberOfTypes=numberOfTypes)
print("done loading data!") 

                        
    #                         
    #                         imageList.append(np.asarray(thresh))
    #                         print("!!!!!!")
    #                         print(thresh)
    #                         print(len(imageList))
                        
#                             img = Image.fromarray(thresh)
#                             img.save('my.png')
#     #                         img.show()
#                             fn = '../output/ETL8G_{:d}_ds{:02d}_{:04d}.png'.format(fileNumber, datasetNumber, characterNumber)
#                             img.save(fn, 'PNG')




