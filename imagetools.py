__author__ = 'Daniel'

import numpy as np

def arraytoimage(a):
    image=np.abs((a*255)/np.amax(a))
    return(image)
