#!/usr/bin/env python3

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import ColorJitter
import os
import sys
import datetime

torch.set_float32_matmul_precision('high')

from models.detector import TextDetectorModel
from dataset.data_fixdata import FixDataDataset
from loss_func import loss_function, CoVWeightingLoss
from dataset.multi import MultiLoader
from dataset.data_detector import get_dataset

lr = 1e-4
EPOCHS = 40
batch=4
logstep=10
iters_to_accumulate=1
output_iter=None
scheduler_gamma = 1.0
continue_train = False
model_size = 'xl'
decoder_only = False

transform = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

class RunningLoss(torch.nn.modules.Module):
    def __init__(self, *args, **kwargs) -> None:
        self.device = kwargs.pop('device', 'cpu')
        self.step = 0
        self.writer = SummaryWriter(log_dir="result2/logs")
        self.runningcount = kwargs.pop('runningcount', 1000)
        self.losses = kwargs.pop('losses', [])
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        self.count = 0
        self.running_loss = {key: torch.tensor(0., dtype=torch.float, device=self.device) for key in self.losses}
        self.correct = torch.tensor(0, device=self.device)
        self.total = torch.tensor(0, device=self.device)

    def write(self, ret=None):
        if ret is None:
            ret = {}
            for key in self.losses:
                ret[key] = self.running_loss[key] / self.count if self.count > 0 else 0.
            ret['accuracy'] = self.correct.float() / self.total.float() if self.total > 0 else torch.tensor(0., dtype=torch.float, device=self.device)

        for key in ret:
            name = 'train/'+key if self.training else 'val/'+key
            self.writer.add_scalar(name, ret[key], self.step)

        return ret

    def forward(self, losses):
        if self.training:
            self.step += 1
        self.count += 1
        for key in self.losses:
            self.running_loss[key] += losses[key]
        self.correct += losses['correct']
        self.total += losses['total']

        ret = {}
        for key in self.losses:
            ret[key] = self.running_loss[key] / self.count if self.count > 0 else 0.
        ret['accuracy'] = self.correct.float() / self.total.float() if self.total > 0 else torch.tensor(0., dtype=torch.float, device=self.device)
        if 'lr' in losses:
            ret['lr'] = losses['lr']

        if self.training and self.count % self.runningcount == 0:
            self.write(ret)
            self.reset()

        return ret

def train():
    if continue_train:
        from load_object import download
        download()

    training_dataset = FixDataDataset('train_data2', 1000)
    training_loader = DataLoader(training_dataset, batch_size=batch, shuffle=True, num_workers=8)

    training_dataset2 = get_dataset(train=True)
    training_loader2 = MultiLoader(training_dataset2.batched(batch, partial=False), workers=8)

    validation_dataset = FixDataDataset('train_data2', 100)
    validation_loader = DataLoader(validation_dataset, batch_size=batch, num_workers=8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using device:', device, flush=True)
    with open('log.txt','w') as wf:
        print(datetime.datetime.now(), 'using device:', device, file=wf, flush=True)

    model = TextDetectorModel(model_size=model_size)
    if os.path.exists('result2/model.pt'):
        data = torch.load('result2/model.pt', map_location="cpu", weights_only=True)
        model.load_state_dict(data['model_state_dict'])
    model.to(device)

    if decoder_only:
        for param in model.detector.parameters():
            param.requires_grad_(False)
        model.detector.eval()

    all_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.RAdam(all_params, lr=lr)
    if 0 < scheduler_gamma < 1.0:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)

    CoWloss = CoVWeightingLoss(momentum=1/1000, device=device, losses=[
        'keymap_loss',
        'size_loss',
        'textline_loss',
        'separator_loss',
        'id_loss',
        *['code%d_loss'%2**(i) for i in range(4)],
    ])
    running_loss = RunningLoss(device=device, runningcount=100, losses=[
        'loss',
        'CoWloss',
        'keymap_loss',
        'size_loss',
        'textline_loss',
        'separator_loss',
        'id_loss',
        *['code%d_loss'%2**(i) for i in range(4)],
    ])

    @torch.compile
    def train_step(image, map, idmap, fmask):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            heatmap, decoder_outputs = model(image, fmask)
            rawloss = loss_function(fmask, map, idmap, heatmap, decoder_outputs)
            loss = CoWloss(rawloss)
        return loss, rawloss

    @torch.compile
    def test_step(image, map, idmap, fmask):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            heatmap, decoder_outputs = model(image, fmask)
            rawloss = loss_function(fmask, map, idmap, heatmap, decoder_outputs)
            loss = CoWloss(rawloss)
        return loss, rawloss

    print('model', model_size, flush=True)
    print('batch', batch, flush=True)
    print('logstep', logstep, flush=True)
    print('lr', lr, flush=True)
    with open('log.txt','a') as wf:
        print('model', model_size, file=wf, flush=True)
        print('batch', batch, file=wf, flush=True)
        print('logstep', logstep, file=wf, flush=True)
        print('lr', lr, file=wf, flush=True)

    last_epoch = 0
    fmask = None
    for epoch in range(last_epoch, EPOCHS):
        print(datetime.datetime.now(), 'epoch', epoch, flush=True)
        print(datetime.datetime.now(), 'lr', optimizer.param_groups[0]['lr'], flush=True)
        with open('log.txt','a') as wf:
            print(datetime.datetime.now(), 'epoch', epoch, file=wf, flush=True)
            print(datetime.datetime.now(), 'lr', optimizer.param_groups[0]['lr'], file=wf, flush=True)

        model.train()
        CoWloss.train()
        running_loss.train()
        if decoder_only:
            model.detector.eval()

        optimizer.zero_grad()
        base_loader = iter(training_loader2)
        for i, data in enumerate(training_loader):
            image, labelmap, idmap = data
            image = image.to(device=device)
            labelmap = labelmap.to(device=device)
            idmap = idmap.to(dtype=torch.long, device=device)

            fmask = model.get_fmask(labelmap, fmask)
            image = transform(image)
            loss, rawloss = train_step(image, labelmap, idmap, fmask)
            scale_loss = loss / iters_to_accumulate
            scale_loss.backward()

            image, labelmap, idmap = next(base_loader)
            image = torch.tensor(image, dtype=torch.float, device=device)
            labelmap = torch.tensor(labelmap, dtype=torch.float, device=device)
            idmap = torch.tensor(idmap, dtype=torch.long, device=device)

            fmask = model.get_fmask(labelmap, fmask)
            image = transform(image)
            loss, rawloss = train_step(image, labelmap, idmap, fmask)
            scale_loss = loss / iters_to_accumulate
            scale_loss.backward()

            if (i + 1) % iters_to_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Gather data and report
            rawloss['CoWloss'] = loss
            rawloss['lr'] = optimizer.param_groups[0]['lr']
            losslog = running_loss(rawloss)
            if (i + 1) % logstep == 0 or i == 0:
                CoW_value = losslog['CoWloss'].item()
                loss_value = losslog['loss'].item()
                acc_value = losslog['accuracy'].item()
                print(epoch, i+1, datetime.datetime.now(), 'CoW', CoW_value, 'loss', loss_value, 'acc', acc_value, flush=True)
                with open('log.txt','a') as wf:
                    print(epoch, i+1, datetime.datetime.now(), 'CoW', CoW_value, 'loss', loss_value, 'acc', acc_value, file=wf, flush=True)

            if output_iter is not None and (i + 1) % output_iter == 0:
                torch.save({
                    'epoch': epoch,
                    'step': i,
                    'model_state_dict': model.state_dict(),
                    }, 'result2/model.pt')

        CoW_value = losslog['CoWloss'].item()
        loss_value = losslog['loss'].item()
        acc_value = losslog['accuracy'].item()
        print(epoch, i+1, datetime.datetime.now(), 'CoW', CoW_value, 'loss', loss_value, 'acc', acc_value, flush=True)
        with open('log.txt','a') as wf:
            print(epoch, i+1, datetime.datetime.now(), 'CoW', CoW_value, 'loss', loss_value, 'acc', acc_value, file=wf, flush=True)

        running_loss.reset()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            }, 'result2/model.pt')

        model.eval()
        CoWloss.eval()
        running_loss.eval()

        with torch.no_grad():
            for vdata in validation_loader:
                image, labelmap, idmap = vdata
                image = image.to(device=device)
                labelmap = labelmap.to(device=device)
                idmap = idmap.to(dtype=torch.long, device=device)

                fmask = model.get_fmask(labelmap, fmask)
                loss, rawloss = test_step(image, labelmap, idmap, fmask)

                rawloss['CoWloss'] = loss
                running_loss(rawloss)

        losslog = running_loss.write()

        CoW_value = losslog['CoWloss'].item()
        loss_value = losslog['loss'].item()
        acc_value = losslog['accuracy'].item()
        print(epoch, 'val', datetime.datetime.now(), 'CoW', CoW_value, 'loss', loss_value, 'acc', acc_value, flush=True)
        with open('log.txt','a') as wf:
            print(epoch, 'val', datetime.datetime.now(), 'CoW', CoW_value, 'loss', loss_value, 'acc', acc_value, file=wf, flush=True)

        running_loss.reset()

        if 0 < scheduler_gamma < 1.0:
            scheduler.step() 


if __name__=='__main__':
    if len(sys.argv) > 1:
        argv = sys.argv[1:]
        for arg in argv:
            if arg.startswith('--epoch'):
                EPOCHS = int(arg.split('=')[1])
            elif arg.startswith('--accumulate'):
                iters_to_accumulate = int(arg.split('=')[1])
            elif arg.startswith('--lr'):
                lr = float(arg.split('=')[1])
            elif arg.startswith('--logstep'):
                logstep = int(arg.split('=')[1])
            elif arg.startswith('--output'):
                output_iter = int(arg.split('=')[1])
            elif arg.startswith('--gamma'):
                scheduler_gamma = float(arg.split('=')[1])
            elif arg.startswith('--continue'):
                continue_train = arg.split('=')[1].lower() == 'true'
            elif arg.startswith('--model'):
                model_size = arg.split('=')[1].lower()
            elif arg.startswith('--decoder'):
                decoder_only = arg.split('=')[1].lower() == 'true'
            else:
                batch = int(arg)

    train()
