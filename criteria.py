import torch
import numpy as np
from PIL import Image, ImageFilter

from models.detector import TextDetectorModel
from util_func import width, height, sigmoid

check_list = {
    'testdata/test1.png': [
        {
            'key': 'key',
            'x': slice(0,70),
            'y': slice(17,150),
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'key',
            'x': slice(60,90),
            'y': slice(20,35),
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'key',
            'x': slice(50,85),
            'y': slice(120,135),
            'direction': 'low',
            'min': 0.1,
        },
    ],
    'testdata/clip15.png': [
        {
            'key': 'code2',
            'x': slice(102,105),
            'y': slice(83,87),
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'code2',
            'x': slice(100,104),
            'y': slice(68,72),
            'direction': 'high',
            'max': 0.9,
        },
        {
            'key': 'code2',
            'x': slice(100,104),
            'y': slice(98,102),
            'direction': 'high',
            'max': 0.9,
        },
    ],
    'testdata/clip16.png': [
        {
            'key': 'code2',
            'x': slice(21,24),
            'y': slice(121,124),
            'direction': 'high',
            'max': 0.9,
        },
        {
            'key': 'code2',
            'x': slice(21,24),
            'y': slice(165,168),
            'direction': 'high',
            'max': 0.9,
        },
        {
            'key': 'code2',
            'x': slice(43,46),
            'y': slice(177,180),
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'code2',
            'x': slice(209,212),
            'y': slice(216,219),
            'direction': 'high',
            'max': 0.9,
        },
        {
            'key': 'code2',
            'x': slice(22,24),
            'y': slice(107,110),
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'code2',
            'x': slice(24,28),
            'y': slice(34,38),
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'code2',
            'x': slice(22,27),
            'y': slice(6,10),
            'direction': 'low',
            'min': 0.1,
        },
    ],
    'testdata/clip17.png': [
        {
            'key': 'key',
            'x': 135,
            'y': 83,
            'direction': 'high',
            'max': 0.9,
        },
        {
            'key': 'code2',
            'x': 126,
            'y': 88,
            'direction': 'high',
            'max': 0.9,
        },
        {
            'key': 'code2',
            'x': 127,
            'y': 76,
            'direction': 'high',
            'max': 0.9,
        },
        {
            'key': 'code2',
            'x': 126,
            'y': 99,
            'direction': 'high',
            'max': 0.9,
        },
        {
            'key': 'code2',
            'x': 126,
            'y': 111,
            'direction': 'high',
            'max': 0.9,
        },
    ],
    'testdata/clip18.png': [
        {
            'key': 'key',
            'x': 31,
            'y': 203,
            'direction': 'high',
            'max': 0.9,
        },
    ],
    'testdata/clip19.png': [
        {
            'key': 'code8',
            'x': 43,
            'y': 37,
            'direction': 'high',
            'max': 0.9,
        },
        {
            'key': 'code8',
            'x': slice(109,112),
            'y': slice(35,38),
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'code8',
            'x': slice(90,94),
            'y': slice(35,38),
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'key',
            'x': slice(220,230),
            'y': slice(134,142),
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'code8',
            'x': 176,
            'y': 34,
            'direction': 'high',
            'max': 0.9,
        },
        {
            'key': 'key',
            'x': 83,
            'y': 37,
            'direction': 'high',
            'max': 0.9,
        },
        {
            'key': 'key',
            'x': 100,
            'y': 36,
            'direction': 'high',
            'max': 0.9,
        },
    ],
    'testdata/clip20.png': [
        {
            'key': 'key',
            'x': 26,
            'y': 42,
            'direction': 'high',
            'max': 0.9,
        },
        {
            'key': 'key',
            'x': slice(117,127),
            'y': slice(83,93),
            'direction': 'low',
            'min': 0.1,
        },
    ],
    'testdata/clip21.png': [
        {
            'key': 'key',
            'x': 52,
            'y': 108,
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'key',
            'x': 72,
            'y': 60,
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'key',
            'x': 120,
            'y': 88,
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'key',
            'x': 211,
            'y': 173,
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'key',
            'x': 203,
            'y': 81,
            'direction': 'high',
            'max': 0.9,
        },
        {
            'key': 'key',
            'x': 199,
            'y': 85,
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'code2',
            'x': 39,
            'y': 79,
            'direction': 'low',
            'min': 0.1,
        },
    ],
    'testdata/clip22.png': [
        {
            'key': 'key',
            'x': slice(81,92),
            'y': slice(23,27),
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'key',
            'x': 100,
            'y': 52,
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'key',
            'x': 126,
            'y': 66,
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'key',
            'x': slice(180,200),
            'y': slice(50,70),
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'code2',
            'x': slice(118,120),
            'y': slice(111,113),
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'code2',
            'x': slice(118,120),
            'y': slice(143,146),
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'code2',
            'x': slice(117,119),
            'y': slice(127,129),
            'direction': 'high',
            'max': 0.9,
        },
    ],
    'testdata/clip23.png': [
        {
            'key': 'key',
            'x': slice(116,127),
            'y': slice(80,90),
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'key',
            'x': slice(164,169),
            'y': slice(75,80),
            'direction': 'low',
            'min': 0.1,
        },
    ],
    'testdata/clip24.png': [
        {
            'key': 'key',
            'x': slice(87,94),
            'y': slice(70,77),
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'key',
            'x': slice(100,105),
            'y': slice(94,99),
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'key',
            'x': slice(105,132),
            'y': slice(121,191),
            'direction': 'low',
            'min': 0.1,
        },
        {
            'key': 'code2',
            'x': slice(137,141),
            'y': slice(225,231),
            'direction': 'low',
            'min': 0.1,
        },
    ],
}

def run_check(model: TextDetectorModel, verbose=False):
    model.eval()
    device = next(model.parameters()).device
    passcount = 0
    failcount = 0
    im = []
    for key in check_list:
        im0 = Image.open(key).convert('RGB')
        im0 = np.asarray(im0)

        padx = max(0, width - im0.shape[1])
        pady = max(0, height - im0.shape[0])
        im0 = np.pad(im0, [[0,pady],[0,padx],[0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))
        im.append(im0)

    im = (np.stack(im, axis=0) / 255.).astype(np.float32)
    images = torch.from_numpy(im).permute(0,3,1,2).to(device=device)
    with torch.no_grad():
        heatmap, features = model.detector(images)
        heatmap = heatmap.cpu().numpy()

    for i,key in enumerate(check_list):
        if verbose:
            print(key)
        target = check_list[key]

        keymap_p = sigmoid(heatmap[i,0,:,:])
        code_p = []
        for k in range(4):
            code_p.append(sigmoid(heatmap[i,5+k,:,:]))

        for criteria in target:
            if criteria['key'] == 'key':
                target_map = keymap_p
            elif criteria['key'] == 'code1':
                target_map = code_p[0]
            elif criteria['key'] == 'code2':
                target_map = code_p[1]
            elif criteria['key'] == 'code4':
                target_map = code_p[2]
            elif criteria['key'] == 'code8':
                target_map = code_p[3]

            x = criteria['x']
            y = criteria['y']
            target = target_map[y,x]

            if 'max' in criteria:
                value = np.max(target)
                if value > criteria['max']:
                    if verbose:
                        print(' PASS', value, criteria)
                    passcount += 1
                else:
                    if verbose:
                        print(' fail', value, criteria)
                    failcount += 1
            elif 'min' in criteria:
                value = np.max(target)
                if value < criteria['min']:
                    if verbose:
                        print(' PASS', value, criteria)
                    passcount += 1
                else:
                    if verbose:
                        print(' fail', value, criteria)
                    failcount += 1
    model.train()
    print('','passcount:',passcount,'failcount',failcount)
    return failcount == 0

if __name__=='__main__':
    model = TextDetectorModel(pre_weights=False)
    data = torch.load('model.pt', map_location="cpu", weights_only=True)
    model.load_state_dict(data['model_state_dict'])

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    device = torch.device(device)
    model.to(device=device)

    run_check(model)
