import numpy as np

def pinned_bw(x, c, b_max):

    def time(x, c, b_max):
        return c + x / b_max

    return x / time(x, c, b_max)


def pageable_bw(x, c, b_max, b_cache, cache_size, b_mem):

    def copy_time(x, b_cache, cache_size, b_mem):
        copy_time = x / b_cache
        if x > 2 ** cache_size:
            copy_time += x / b_mem
        return copy_time

    def time(x, c, b_max, b_cache, cache_size,  b_mem):
        t_c = np.vectorize(copy_time)(x, b_cache, cache_size, b_mem)
        t_l = x / b_max

        return c + t_c + t_l 

    return x / time(x, c, b_max, b_cache, cache_size, b_mem)


def minsky_pageable_bw(bytes, overhead, nvlink_bw, l1_bw, l1_size, l2_bw, l2_size, l3_bw, l3_size, m_bw):

    def nvlink_time(bytes, nvlink_bw):
        return bytes / nvlink_bw

    def m_time(bytes, l1_bw, l1_size, l2_bw, l2_size, l3_bw, l3_size, m_bw):
        if bytes < l1_size:
            time = bytes / l1_bw
        elif bytes < l2_size:
            time = bytes / l2_bw
        elif bytes < l3_size:
            time = bytes / l3_bw
        else:
            time = bytes / m_bw
        return time

    def time(bytes, overhead, nvlink_bw, l1_bw, l1_size, l2_bw, l2_size, l3_bw, l3_size, m_bw):
        return np.maximum(nvlink_time(bytes, nvlink_bw), np.vectorize(m_time)(bytes, l1_bw, l1_size, l2_bw, l2_size, l3_bw, l3_size, m_bw)) + overhead

    return bytes / time(bytes, overhead, nvlink_bw, l1_bw, l1_size, l2_bw, l2_size, l3_bw, l3_size, m_bw)
