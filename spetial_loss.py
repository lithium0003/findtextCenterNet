import torch
from torchvision.transforms import ColorJitter
import numpy as np
from PIL import Image, ImageFilter
import glob

from models.detector import TextDetectorModel
from util_func import width, height
from criteria import check_list

def sp_lossfunc(model: TextDetectorModel):
    # @torch.compile
    def sptrain_step(images):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            heatmap, features = model.detector(images)
        return heatmap

    model.eval()
    transform = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    device = next(model.parameters()).device
    for key in reversed(check_list):
        files = sorted(glob.glob(key))
        for file in files:
            im0 = Image.open(file).convert('RGB')
            im0 = np.asarray(im0)

            im_ave = np.median(im0, axis=(0,1), keepdims=True)

            if im0.shape[1] > width:
                offsetx = np.random.randint(im0.shape[1] - width)
                im0 = im0[:,offsetx:]
            if im0.shape[0] > height:
                offsety = np.random.randint(im0.shape[0] - height)
                im0 = im0[offsety:,:]

            if im0.shape[1] > width:
                im0 = im0[:,:width]
            if im0.shape[0] > height:
                im0 = im0[:height,:]

            padx = max(0, width - im0.shape[1])
            pady = max(0, height - im0.shape[0])
            offsetx = np.random.randint(padx) if padx > 0 else 0
            offsety = np.random.randint(pady) if pady > 0 else 0
            padx -= offsetx
            pady -= offsety
            im1 = np.pad(im0, [[offsety,pady],[offsetx,padx],[0,0]], 'constant', constant_values=((255,255),(255,255),(255,255)))

            im2 = np.empty_like(im1)
            im2[:,:,:] = im_ave
            im2[offsety:-(pady+1),offsetx:-(padx+1),:] = im1[offsety:-(pady+1),offsetx:-(padx+1),:]
            im1 = im2

            offsetx = (offsetx + 2) // 4
            offsety = (offsety + 2) // 4

            im1 = (im1 / 255.).astype(np.float32)
            im = np.expand_dims(im1, 0)

            images = torch.from_numpy(im).permute(0,3,1,2).to(device=device)
            images = transform(images)

            heatmap = sptrain_step(images)

            loss = None
            keymap = heatmap[0,0,:,:]
            textline = heatmap[0,3,:,:]
            sepline = heatmap[0,4,:,:]
            code = []
            for k in range(4):
                code.append(heatmap[0,5+k,:,:])

            for criteria in check_list[key]:
                if criteria['key'] == 'key':
                    target_map = keymap
                elif criteria['key'] == 'code1':
                    target_map = code[0]
                elif criteria['key'] == 'code2':
                    target_map = code[1]
                elif criteria['key'] == 'code4':
                    target_map = code[2]
                elif criteria['key'] == 'code8':
                    target_map = code[3]
                elif criteria['key'] == 'textline':
                    target_map = textline
                elif criteria['key'] == 'sepline':
                    target_map = sepline

                x = criteria['x']
                y = criteria['y']
                if isinstance(x, slice):
                    x = slice(x.start + offsetx, x.stop + offsetx if x.stop is not None else None)
                else:
                    x += offsetx
                if isinstance(y, slice):
                    y = slice(y.start + offsety, y.stop + offsety if y.stop is not None else None)
                else:
                    y += offsety
                target = target_map[y,x]

                weight = criteria.get('weight', 1.0)

                if criteria['direction'] == 'high':
                    with torch.no_grad():
                        value = torch.nn.functional.sigmoid(target.float()).max().cpu().numpy()
                    if value > criteria['threshold']:
                        continue
                    else:
                        mask = torch.nn.functional.sigmoid(target.float()) <= criteria['threshold']
                        masked_target = torch.masked_select(target, mask)
                        if loss is None:
                            loss = -torch.masked.mean(torch.nn.functional.logsigmoid(masked_target)) * weight
                        else:
                            loss += -torch.masked.mean(torch.nn.functional.logsigmoid(masked_target)) * weight
                elif criteria['direction'] == 'low':
                    with torch.no_grad():
                        value = torch.nn.functional.sigmoid(target.float()).max().cpu().numpy()
                    if value < criteria['threshold']:
                        continue
                    else:
                        mask = torch.nn.functional.sigmoid(target.float()) >= criteria['threshold']
                        masked_target = torch.masked_select(target, mask)
                        if loss is None:
                            loss = torch.masked.mean(masked_target + torch.nn.functional.softplus(-masked_target)) * weight
                        else:
                            loss += torch.masked.mean(masked_target + torch.nn.functional.softplus(-masked_target)) * weight

            if loss is not None:
                loss.backward()

    model.train()

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

    sp_lossfunc(model)
