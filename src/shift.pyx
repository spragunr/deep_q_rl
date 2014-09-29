from numpy cimport ndarray as ar
import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.uint8


@cython.boundscheck(False)
@cython.wraparound(False)
def shift3d_uint8(ar[np.uint8_t,ndim=3] data, int shift_amt):
    cdef int i, j, k
    cdef int n = data.shape[0]
    cdef int h = data.shape[1]
    cdef int w = data.shape[2]
    
    for i in xrange(shift_amt, n):
        for j in xrange(h):
            for k in xrange(w):
                data[i-shift_amt, j, k] = data[i, j, k]

@cython.boundscheck(False)
@cython.wraparound(False)
def shift1d_float32(ar[np.float32_t,ndim=1] data, int shift_amt):
    cdef int i
    cdef int n = data.shape[0]
    for i in xrange(shift_amt, n):
        data[i-shift_amt] = data[i]


@cython.boundscheck(False)
@cython.wraparound(False)
def shift1d_int32(ar[np.int32_t,ndim=1] data, int shift_amt):
    cdef int i
    cdef int n = data.shape[0]
    for i in xrange(shift_amt, n):
        data[i-shift_amt] = data[i]


