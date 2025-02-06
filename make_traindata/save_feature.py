import numpy as np
import glob
import os

def load():
    path = 'code_features'
    data = {}
    for filename in sorted(glob.glob(os.path.join(path,'*.npy'))):
        print(filename)
        codestr = os.path.splitext(os.path.basename(filename))[0]
        horizontal = codestr[0] == 'h'
        code = int(codestr[1:], 16)

        value = np.load(filename).astype(np.float16)
        if horizontal:
            data['hori_%d'%code] = value
        else:
            data['vert_%d'%code] = value
    np.savez('features', **data)

if __name__=='__main__':
    load()
