import numpy as np
import sys
import glob
import os

def load(path):
    base_path = 'code_features'
    os.makedirs(base_path, exist_ok=True)
    for filename in sorted(glob.glob(os.path.join(path,'*.npy'))):
        print(filename)
        value = np.load(filename)
        basename = os.path.basename(filename)

        filename2 = os.path.join(base_path, basename)
        if os.path.exists(filename2):
            data2 = np.load(filename2)
            value = np.concatenate([value,data2], axis=0)
        np.save(filename2, value)

if __name__=='__main__':
    for path in sys.argv[1:]:
        load(path)
