# -*- coding:utf-8 -*-
from __future__ import absolute_import

from script.process import util
from script.config import config

import pandas as pd
import gc

total_size = config.total_size
path = config.path

chunk_size = 2000000
iters = 10

batch = total_size / (chunk_size * iters)
rest = total_size % (chunk_size * iters)

reader = pd.read_table(path, sep=',', engine='c', iterator=True,
                       dtype={'file_id': 'int32', 'label': 'int8', 'index': 'int16'})
batch = 2
for i in range(batch):
    train_chunk = util.readchunks(reader, chunk_size, iters)
    files_cnt1 = train_chunk.file_id.value_counts()
    files_label1 = train_chunk[['file_id', 'label']].drop_duplicates()
    if i == 0:
        sums = files_cnt1
        labels = files_label1
    else:
        sums = util.countFiles(sums, files_cnt1)
        labels = util.labelFiles(labels, files_label1)

    del train_chunk
    del files_cnt1
    del files_label1

sums = sums.to_frame().reset_index().rename(columns={'index': 'file_id', 0: 'file_cnt'})

sums = sums.merge(labels,on = ['file_id'])