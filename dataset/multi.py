#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""An alternative to DataLoader using ZMQ.

This implements MultiLoader, an alternative to DataLoader when torch
is not available. Subprocesses communicate with the loader through
ZMQ, provided for high performance multithreaded queueing.
"""

import multiprocessing as mp
import os
import pickle
import uuid
import weakref
import time

import zmq

the_protocol = pickle.HIGHEST_PROTOCOL

all_pids = weakref.WeakSet()


class EOF:
    """A class that indicates that a data stream is finished."""

    def __init__(self, **kw):
        """Initialize the class with the kw as instance variables."""
        self.__dict__.update(kw)


def reader(dataset, sockname1, sockname2, index, num_workers):
    """Read samples from the dataset and send them over the socket.

    :param dataset: source dataset
    :param sockname: name for the socket to send data to
    :param index: index for this reader, using to indicate EOF
    """
    global the_protocol
    os.environ["WORKER"] = str(index)
    os.environ["NUM_WORKERS"] = str(num_workers)
    ctx = zmq.Context.instance()
    sock1 = ctx.socket(zmq.PUSH)
    sock1.connect(sockname1)
    sock2 = ctx.socket(zmq.SUB)
    sock2.connect(sockname2)
    sock2.setsockopt(zmq.SUBSCRIBE, b'')
    poller = zmq.Poller()
    poller.register(sock2, zmq.POLLIN)
    rcount = 0
    for i, sample in enumerate(dataset):
        data = pickle.dumps(sample, protocol=the_protocol)
        sock1.send(data)
        while True:
            socks = dict(poller.poll(100))
            if sock2 in socks and socks[sock2] == zmq.POLLIN:
                rcount = sock2.recv_pyobj()
            if i > rcount / num_workers + 2:
                time.sleep(0.1)
            else:
                break
    sock1.send(pickle.dumps(EOF(index=index)))
    sock1.close()


class MultiLoader:
    """Alternative to PyTorch DataLoader based on ZMQ."""

    def __init__(
        self, dataset, workers=4, verbose=False, nokill=False, prefix="/tmp/_multi-"
    ):
        """Create a MultiLoader for a dataset.

        This creates ZMQ sockets, spawns `workers` subprocesses, and has them send data
        to the socket.

        :param dataset: source dataset
        :param workers: number of workers
        :param verbose: report progress verbosely
        :param nokill: don't kill old processes when restarting (allows multiple loaders)
        :param prefix: directory prefix for the ZMQ socket
        """
        self.dataset = dataset
        self.workers = workers
        self.verbose = verbose
        self.pids = []
        self.socket1 = None
        self.socket2 = None
        self.ctx = zmq.Context.instance()
        self.nokill = nokill
        self.prefix = prefix

    def kill(self):
        """kill."""
        for pid in self.pids:
            if pid is None:
                continue
            if self.verbose:
                print("killing", pid)
            pid.kill()
            pid.join(1.0)
        self.pids = []
        if self.socket1 is not None:
            if self.verbose:
                print("closing", self.socket1)
            self.socket1.close()
        self.socket1 = None
        if self.socket2 is not None:
            if self.verbose:
                print("closing", self.socket2)
            self.socket2.close()
        self.socket2 = None

    def __iter__(self):
        """Return an iterator over this dataloader."""
        if not self.nokill:
            self.kill()
        self.sockname1 = "ipc://" + self.prefix + str(uuid.uuid4())
        self.sockname2 = "ipc://" + self.prefix + str(uuid.uuid4())
        self.socket1 = self.ctx.socket(zmq.PULL)
        self.socket1.bind(self.sockname1)
        if self.verbose:
            print("#", self.sockname1)
        self.socket2 = self.ctx.socket(zmq.PUB)
        self.socket2.bind(self.sockname2)
        if self.verbose:
            print("#", self.sockname2)
        self.pids = [None] * self.workers
        for index in range(self.workers):
            args = (self.dataset, self.sockname1, self.sockname2, index, self.workers)
            self.pids[index] = mp.Process(target=reader, args=args, daemon=True)
        all_pids.update(self.pids)
        for pid in self.pids:
            pid.start()
        count = 0
        self.socket2.send_pyobj(count)
        while self.pids.count(None) < len(self.pids):
            data = self.socket1.recv()
            sample = pickle.loads(data)
            if isinstance(sample, EOF):
                if self.verbose:
                    print("# subprocess finished", sample.index)
                self.pids[sample.index].join(1.0)
                self.pids[sample.index] = None
            else:
                yield sample
            count += 1
            self.socket2.send_pyobj(count)
