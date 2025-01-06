import torch
import glob
import os
import json
from PIL import Image
import numpy as np

from .processer import process2
Image.MAX_IMAGE_PIXELS = 1000000000

rng = np.random.default_rng()

class FixDataDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, repeat_count):
        super().__init__()
        self.data_path = data_path
        self.repeat_count = repeat_count
    
        self.jsonfiles = sorted(glob.glob(os.path.join(data_path, '*.json')))
        self.imagefiles = [os.path.splitext(f)[0] for f in self.jsonfiles]
        self.sepsfiles = [f+'.seps.png' for f in self.imagefiles]
        self.linesfiles = [f+'.lines.png' for f in self.imagefiles]

        self.jsons = []
        for jsonfile in self.jsonfiles:
            with open(jsonfile, 'r', encoding='utf-8') as file:
                self.jsons.append(json.load(file))

        self.positions = []
        self.codelists = []
        for data in self.jsons:
            position = np.zeros(shape=(len(data['textbox']), 4), dtype=np.float32)
            codelist = np.zeros(shape=(len(data['textbox']), 2), dtype=np.int32)
            for i, pos in enumerate(data['textbox']):
                cx = pos['cx']
                cy = pos['cy']
                w = pos['w']
                h = pos['h']
                position[i,0] = cx
                position[i,1] = cy
                position[i,2] = w
                position[i,3] = h
                text = pos['text']
                if text is not None:
                    c = int.from_bytes(text.encode("utf-32-le"), byteorder='little')
                else:
                    c = 0
                code1 = 1 if pos['p_code1'] > 0.5 else 0
                code2 = 2 if pos['p_code2'] > 0.5 else 0
                code4 = 4 if pos['p_code4'] > 0.5 else 0
                code8 = 8 if pos['p_code8'] > 0.5 else 0
                code = code1 + code2 + code4 + code8
                codelist[i,0] = c
                codelist[i,1] = code
            self.positions.append(position)
            self.codelists.append(codelist)

    def __len__(self):
        return len(self.jsonfiles) * self.repeat_count

    def __getitem__(self, idx):
        idx = idx % len(self.jsonfiles)

        im0 = np.asarray(Image.open(self.imagefiles[idx]).convert('RGB'))
        seps = np.asarray(Image.open(self.sepsfiles[idx]))
        lines = np.asarray(Image.open(self.linesfiles[idx]))
        posision = self.positions[idx]
        codelist = self.codelists[idx]

        image, mapimage, indexmap = process2(im0, lines, seps, posision, codelist)
        return image, mapimage, indexmap

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision.transforms import ColorJitter

    transform = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

    dataset = FixDataDataset('train_data2',100)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for sample in dataloader:
        image, labelmap, idmap = sample
        image = transform(image)

        plt.figure()
        if len(image[0].shape) > 2:
            plt.imshow(image[0].permute(1,2,0))
        else:
            plt.imshow(image[0])

        plt.figure()
        plt.subplot(2,4,1)
        if len(image[0].shape) > 2:
            plt.imshow(image[0].permute(1,2,0))
        else:
            plt.imshow(image[0])
        for i in range(5):
            plt.subplot(2,4,2+i)
            plt.imshow(labelmap[0,i])
        plt.subplot(2,4,7)
        plt.imshow(idmap[0,0])
        plt.subplot(2,4,8)
        plt.imshow(idmap[0,1])
        plt.show()
