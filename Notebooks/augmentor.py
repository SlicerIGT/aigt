import numpy as np
from multiprocessing import Queue

import scipy
import scipy.ndimage
import cv2

def prepare_batches(y_edge, data, q):
    while True:
        angle = np.random.randint(-17, 17)   
        batchx = scipy.ndimage.interpolation.rotate(data, angle, (1, 2), False, mode="constant", cval=0)
        batchy = scipy.ndimage.interpolation.rotate(y_edge, angle, (1, 2), False, mode="constant", cval=-1)
        
        batchx = np.clip(batchx, 0, 1)

        #batchx = np.concatenate([batchx, -batchx]) / 255.

        maxcrop = 40
        shape = batchx[0].shape
        
        final_x = []
        final_y = []
        for i in range(len(batchx)):
            
            while True:
                
                x1 = np.random.randint(0, maxcrop)
                x2 = np.random.randint(256 - maxcrop, 256)

                y1 = np.random.randint(0, maxcrop)
                y2 = np.random.randint(256 - maxcrop, 256)

                y_test = batchy[i, x1:x2, y1:y2]

                if not(np.any(y_test == -1)):
                    break
            
            final_x.append(cv2.resize(batchx[i, x1:x2, y1:y2], (128, 128)))
            final_y.append(cv2.resize(batchy[i, x1:x2, y1:y2], (128, 128)))
            

        
        
        batchx = np.array(final_x)
        batchy = np.array(final_y)
        
        print(batchx.shape)
        print(batchy.shape)
        batchx = np.expand_dims(batchx, -1)

        batchy = np.expand_dims(batchy, -1)
        batchy = np.concatenate([batchy, 1-batchy], -1)
        
        batchx = np.concatenate([batchx, np.flip(batchx, 2)])
        batchy = np.concatenate([batchy, np.flip(batchy, 2)])
        
        #batchx = np.concatenate([batchx, np.flip(batchx, 1)])
        #batchy = np.concatenate([batchy, np.flip(batchy, 1)])
        
        #batchx = np.concatenate([batchx, np.transpose(batchx, (0, 2, 1, 3))])
        #batchy = np.concatenate([batchy, np.transpose(batchy, (0, 2, 1, 3))])
        
        #offset = np.random.random((len(batchx), 1, 1, 1)) / 4 - .125
        #scale = np.random.random((len(batchx), 1, 1, 1)) / 2 + .75
 
        #batchx = scale * batchx + offset
    
        scale = np.random.uniform(-1.4, 1.4, (len(batchx), 1, 1, 1))

        batchx = (batchx + 0.0) ** ( 2.5 **scale)
        q.put((batchx, batchy))