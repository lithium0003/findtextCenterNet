#!/usr/bin/env python3

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm
# from torch.utils.data import DataLoader
import os
import sys
import glob
import datetime

torch.set_float32_matmul_precision('high')

from models.detector import TextDetectorModel
from dataset.data_detector import get_dataset
from loss_func import loss_function, CoVWeightingLoss
from dataset.multi import MultiLoader

upload_objectstorage = False

lr = 1e-3
EPOCHS = 40
batch=4
compile=True
logstep=100
iters_to_accumulate=1
output_iter=None
scheduler_gamma = 0.9

class RunningLoss(torch.nn.modules.Module):
    def __init__(self, *args, **kwargs) -> None:
        self.device = kwargs.pop('device', 'cpu')
        self.step = 0
        self.writer = SummaryWriter(log_dir="result1/logs")
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

        if upload_objectstorage:
            self.writer.flush()
            log = sorted(glob.glob('result1/logs/*'))
            if len(log) > 0:
                upload(log[0], 'logs')

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
    training_dataset = get_dataset(train=True)
    # training_loader = DataLoader(training_dataset, batch_size=batch, num_workers=4)
    training_loader = MultiLoader(training_dataset.batched(batch, partial=False), workers=8)

    validation_dataset = get_dataset(train=False)
    # validation_loader = DataLoader(validation_dataset, batch_size=batch, num_workers=4)
    validation_loader = MultiLoader(validation_dataset.batched(batch, partial=False), workers=8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using device:', device, flush=True)
    with open('log.txt','w') as wf:
        print(datetime.datetime.now(), 'using device:', device, file=wf, flush=True)
    if upload_objectstorage:
        upload('log.txt', 'log.txt')

    model = TextDetectorModel()
    if os.path.exists('result1/model.pt'):
        data = torch.load('result1/model.pt', map_location="cpu", weights_only=True)
        model.load_state_dict(data['model_state_dict'])
    model.to(device)

    all_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam([
        {'params': all_params},
    ], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)

    CoWloss = CoVWeightingLoss(momentum=1/100, device=device, losses=[
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
        'l2_loss',
        *['code%d_loss'%2**(i) for i in range(4)],
    ])

    def train_step(image, map, idmap, fmask):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            heatmap, decoder_outputs, features = model(image, fmask)
            rawloss = loss_function(fmask, map, idmap, heatmap, decoder_outputs, features)
            loss = CoWloss(rawloss) + rawloss['l2_loss']
        return loss, rawloss

    def test_step(image, map, idmap, fmask):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            heatmap, decoder_outputs, features = model(image, fmask)
            rawloss = loss_function(fmask, map, idmap, heatmap, decoder_outputs, features)
            loss = CoWloss(rawloss) + rawloss['l2_loss']
        return loss, rawloss

    if compile:
        print('compile', flush=True)
        with open('log.txt','a') as wf:
            print('compile', file=wf, flush=True)
        if upload_objectstorage:
            upload('log.txt', 'log.txt')
        train_step = torch.compile(train_step)
        test_step = torch.compile(test_step)
    else:
        print('no compile', flush=True)
        with open('log.txt','a') as wf:
            print('no compile', file=wf, flush=True)
        if upload_objectstorage:
            upload('log.txt', 'log.txt')

    print('batch', batch, flush=True)
    print('logstep', logstep, flush=True)
    print('lr', lr, flush=True)
    with open('log.txt','a') as wf:
        print('batch', batch, file=wf, flush=True)
        print('logstep', logstep, file=wf, flush=True)
        print('lr', lr, file=wf, flush=True)
    if upload_objectstorage:
        upload('log.txt', 'log.txt')

    last_epoch = 0
    fmask = None
    for epoch in range(last_epoch, EPOCHS):
        print(datetime.datetime.now(), 'epoch', epoch, flush=True)
        print(datetime.datetime.now(), 'lr', optimizer.param_groups[0]['lr'], flush=True)
        with open('log.txt','a') as wf:
            print(datetime.datetime.now(), 'epoch', epoch, file=wf, flush=True)
            print(datetime.datetime.now(), 'lr', optimizer.param_groups[0]['lr'], file=wf, flush=True)
        if upload_objectstorage:
            upload('log.txt', 'log.txt')

        model.train()
        CoWloss.train()
        running_loss.train()

        optimizer.zero_grad()
        for i, data in enumerate(training_loader):
            image, labelmap, idmap = data
            # image = image.to(device=device)
            # labelmap = labelmap.to(device=device)
            # idmap = idmap.to(dtype=torch.long, device=device)
            image = torch.tensor(image, dtype=torch.float, device=device)
            labelmap = torch.tensor(labelmap, dtype=torch.float, device=device)
            idmap = torch.tensor(idmap, dtype=torch.long, device=device)

            fmask = model.get_fmask(labelmap, fmask)
            loss, rawloss = train_step(image, labelmap, idmap, fmask)
            loss.backward()
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

                if upload_objectstorage:
                    upload('log.txt', 'log.txt')
            
            if output_iter is not None and (i + 1) % output_iter == 0:
                torch.save({
                    'epoch': epoch,
                    'step': i,
                    'model_state_dict': model.state_dict(),
                    }, 'result1/model.pt')

        CoW_value = losslog['CoWloss'].item()
        loss_value = losslog['loss'].item()
        acc_value = losslog['accuracy'].item()
        print(epoch, i+1, datetime.datetime.now(), 'CoW', CoW_value, 'loss', loss_value, 'acc', acc_value, flush=True)
        with open('log.txt','a') as wf:
            print(epoch, i+1, datetime.datetime.now(), 'CoW', CoW_value, 'loss', loss_value, 'acc', acc_value, file=wf, flush=True)

        if upload_objectstorage:
            upload('log.txt', 'log.txt')

        running_loss.reset()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            }, 'result1/model.pt')

        if upload_objectstorage:
            upload('result1/model.pt', 'epoch%04d.pt'%epoch)

        model.eval()
        CoWloss.eval()
        running_loss.eval()

        with torch.no_grad():
            for vdata in validation_loader:
                image, labelmap, idmap = vdata
                # image = image.to(device=device)
                # labelmap = labelmap.to(device=device)
                # idmap = idmap.to(dtype=torch.long, device=device)
                image = torch.tensor(image, dtype=torch.float, device=device)
                labelmap = torch.tensor(labelmap, dtype=torch.float, device=device)
                idmap = torch.tensor(idmap, dtype=torch.long, device=device)

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

        if upload_objectstorage:
            upload('log.txt', 'log.txt')

        running_loss.reset()

        scheduler.step() 


if __name__=='__main__':
    if len(sys.argv) > 1:
        argv = sys.argv[1:]
        for arg in argv:
            if arg.startswith('--compile'):
                compile = arg.split('=')[1].lower() == 'true'
            elif arg.startswith('--epoch'):
                EPOCHS = int(arg.split('=')[1])
            elif arg.startswith('--accumulate'):
                iters_to_accumulate = int(arg.split('=')[1])
            elif arg.startswith('--lr'):
                lr = float(arg.split('=')[1])
            elif arg.startswith('--logstep'):
                logstep = int(arg.split('=')[1])
            elif arg.startswith('--upload'):
                upload_objectstorage = arg.split('=')[1].lower() == 'true'
            elif arg.startswith('--output'):
                output_iter = int(arg.split('=')[1])
            elif arg.startswith('--gamma'):
                scheduler_gamma = float(arg.split('=')[1])
            else:
                batch = int(arg)

    if upload_objectstorage:
        from put_object import upload

    train()
