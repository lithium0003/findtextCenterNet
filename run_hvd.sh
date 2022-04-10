#!/bin/bash

horovodrun -np 8 python3 -u train_hvd.py
