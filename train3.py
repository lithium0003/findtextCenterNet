#!/usr/bin/env python3

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
import os
import sys
import datetime
import numpy as np

torch.set_float32_matmul_precision('high')

from models.radam_schedulefree import RAdamScheduleFree
from models.transformer import ModelDimensions, Transformer, TransformerPredictor
from dataset.data_transformer import TransformerDataDataset
from loss_func import loss_function3

EPOCHS = 1000
lr=1e-4
batch=128
logstep=10
output_iter=None
save_all=False

rng = np.random.default_rng()

class RunningLoss(torch.nn.modules.Module):
    def __init__(self, *args, **kwargs) -> None:
        self.device = kwargs.pop('device', 'cpu')
        self.step = 0
        self.writer = SummaryWriter(log_dir="result3/logs")
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
        if self.count == 0:
            return ret

        if ret is None:
            ret = {}
            for key in self.losses:
                ret[key] = self.running_loss[key] / self.count if self.count > 0 else torch.tensor(0., dtype=torch.float, device=self.device)
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
    prep = TransformerDataDataset.prepare()
    last_epoch = 0

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    device = torch.device(device)
    print('using device:', device, flush=True)

    num_workers = os.cpu_count()
    training_dataset = TransformerDataDataset(*prep, train=True)
    training_loader = DataLoader(training_dataset, batch_size=batch, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)

    validation_dataset = TransformerDataDataset(*prep, train=False)
    validation_loader = DataLoader(validation_dataset, batch_size=batch, num_workers=num_workers, drop_last=True, pin_memory=True)

    with open('log.txt','w') as wf:
        print(datetime.datetime.now(), 'using device:', device, file=wf, flush=True)

    if os.path.exists('result3/model.pt'):
        data = torch.load('result3/model.pt', map_location="cpu", weights_only=True)
        config = ModelDimensions(**data['config'])
        model = Transformer(**config.__dict__)
        model.load_state_dict(data['model_state_dict'])
        last_epoch = data['epoch']
    else:
        config = ModelDimensions()
        model = Transformer(**config.__dict__)
    model2 = TransformerPredictor(model.encoder, model.decoder)
    model.to(device)
    model2.to(device)

    all_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = RAdamScheduleFree(all_params, lr=lr)

    running_loss = RunningLoss(device=device, runningcount=100, losses=[
        'loss',
    ])

    @torch.compile
    def train_step(encoder_input, decoder_input, label_code):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(encoder_input, decoder_input)
            rawloss = loss_function3(outputs, label_code)
        return rawloss['loss'], rawloss

    @torch.compile
    def test_step(encoder_input, decoder_input, label_code):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(encoder_input, decoder_input)
            rawloss = loss_function3(outputs, label_code)
        return rawloss['loss'], rawloss

    print('batch', batch, flush=True)
    print('logstep', logstep, flush=True)
    with open('log.txt','a') as wf:
        print('batch', batch, file=wf, flush=True)
        print('logstep', logstep, file=wf, flush=True)
    loader_len = len(training_loader)

    scaler = torch.amp.GradScaler()
    for epoch in range(last_epoch, EPOCHS):
        print(datetime.datetime.now(), 'epoch', epoch, flush=True)
        print(datetime.datetime.now(), 'lr', optimizer.param_groups[0]['lr'], flush=True)
        with open('log.txt','a') as wf:
            print(datetime.datetime.now(), 'epoch', epoch, file=wf, flush=True)
            print(datetime.datetime.now(), 'lr', optimizer.param_groups[0]['lr'], file=wf, flush=True)

        model.train()
        running_loss.train()
        optimizer.train()

        optimizer.zero_grad()
        for i, data in enumerate(training_loader):
            text, feature, in_codes, out_codes = data
            feature = feature.to(dtype=torch.float32, device=device, non_blocking=True)
            in_codes = in_codes.to(device=device, non_blocking=True)
            out_codes = out_codes.to(device=device, non_blocking=True)

            loss, rawloss = train_step(feature, in_codes, out_codes)
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Gather data and report
            rawloss['lr'] = optimizer.param_groups[0]['lr']
            losslog = running_loss(rawloss)

            if (i + 1) % logstep == 0 or i == 0:
                loss_value = losslog['loss'].item()
                acc_value = losslog['accuracy'].item()
                print(epoch, i+1, f'{(i+1)/loader_len:.2%}', datetime.datetime.now(), 'loss', loss_value, 'acc', acc_value, flush=True)
                with open('log.txt','a') as wf:
                    print(epoch, i+1, datetime.datetime.now(), 'loss', loss_value, 'acc', acc_value, file=wf, flush=True)

            if output_iter is not None and (i + 1) % output_iter == 0:
                optimizer.eval()
                if save_all:
                    torch.save({
                        'epoch': epoch,
                        'step': i,
                        'config': config.__dict__,
                        'model_state_dict': model.state_dict(),
                        }, f'result3/model-epoch{epoch:08}-step{i:08}.pt')
                else:
                    torch.save({
                        'epoch': epoch,
                        'step': i,
                        'config': config.__dict__,
                        'model_state_dict': model.state_dict(),
                        }, 'result3/model.pt')

        running_loss.write(losslog)
        loss_value = losslog['loss'].item()
        acc_value = losslog['accuracy'].item()
        print(epoch, i+1, f'{(i+1)/loader_len:.2%}', datetime.datetime.now(), 'loss', loss_value, 'acc', acc_value, flush=True)
        with open('log.txt','a') as wf:
            print(epoch, i+1, datetime.datetime.now(), 'loss', loss_value, 'acc', acc_value, file=wf, flush=True)

        running_loss.reset()

        optimizer.eval()
        if save_all:
            torch.save({
                'epoch': epoch,
                'config': config.__dict__,
                'model_state_dict': model.state_dict(),
                }, f'result3/model-epoch{epoch:08}.pt')
        else:
            torch.save({
                'epoch': epoch,
                'config': config.__dict__,
                'model_state_dict': model.state_dict(),
                }, 'result3/model.pt')

        model.eval()
        running_loss.eval()

        with torch.no_grad():
            for vdata in validation_loader:
                text, feature, in_codes, out_codes = vdata
                feature = feature.to(dtype=torch.float32, device=device, non_blocking=True)
                in_codes = in_codes.to(device=device, non_blocking=True)
                out_codes = out_codes.to(device=device, non_blocking=True)

                loss, rawloss = test_step(feature, in_codes, out_codes)
                running_loss(rawloss)

        losslog = running_loss.write()

        loss_value = losslog['loss'].item()
        acc_value = losslog['accuracy'].item()
        print(epoch, 'val', datetime.datetime.now(), 'loss', loss_value, 'acc', acc_value, flush=True)
        with open('log.txt','a') as wf:
            print(epoch, 'val', datetime.datetime.now(), 'loss', loss_value, 'acc', acc_value, file=wf, flush=True)

        running_loss.reset()

        model2.eval()
        idx = rng.integers(len(validation_dataset))
        with torch.no_grad():
            vdata = validation_dataset[idx]
            text, feature, in_codes, out_codes = vdata
            print('==================')
            print(text)
            feature = torch.tensor(feature[None,:,:], dtype=torch.float32, device=device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model2(feature).squeeze(0).cpu().numpy()
            predstr = ''
            for p in pred:
                if p == 1:
                    continue
                if p == 0 or p == 2:
                    break
                if p < 0x10FFFF:
                    predstr += chr(p)
                else:
                    predstr += '\uFFFD'
            print('------------------')
            try:
                print(predstr)
            except UnicodeEncodeError:
                pass
        running_loss.writer.add_text('true', text, global_step=running_loss.step)
        running_loss.writer.add_text('pred', predstr, global_step=running_loss.step)
        running_loss.writer.flush()

if __name__=='__main__':
    if len(sys.argv) > 1:
        argv = sys.argv[1:]
        for arg in argv:
            if arg.startswith('--epoch'):
                EPOCHS = int(arg.split('=')[1])
            elif arg.startswith('--lr'):
                lr = float(arg.split('=')[1])
            elif arg.startswith('--logstep'):
                logstep = int(arg.split('=')[1])
            elif arg.startswith('--output'):
                output_iter = int(arg.split('=')[1])
            elif arg.startswith('--all'):
                save_all = True
            else:
                batch = int(arg)

    train()
