#!/bin/bash

MPIOPTS="-np 8 -bind-to none -map-by slot -x PATH -mca pml ob1 -mca btl ^openib"
mpirun ${MPIOPTS} python3 -u train.py
