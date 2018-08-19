# -*- coding:utf-8 -*-
import pandas as pd


def readchunks(reader, chunksize, iters):
    train_data = pd.DataFrame()
    loop = True
    for i in range(iters):
        if loop:
            try:
                chunks = reader.get_chunk(chunksize)
                train_data = train_data.append(chunks)
            except StopIteration:
                print("reader is end, cannot get_chunk!")
                loop = False

    return train_data


def countFiles(cnt1, cnt2):
    union_index = cnt1.index.union(cnt2.index)
    sums = pd.Series(0, index=union_index)

    sums =sums.add(cnt1,fill_value = 0)
    sums =sums.add(cnt2,fill_value = 0)
    return sums


def labelFiles(cnt1, cnt2):
    cnt1 = cnt1.append(cnt2,ignore_index = True).drop_duplicates()

    return cnt1
