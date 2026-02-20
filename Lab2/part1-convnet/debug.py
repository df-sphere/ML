import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def run(X, K, s):
    k_size = K.shape[0]

    x_indx = np.arange(np.prod(X.shape)).reshape(X.shape)

    sw_indx = sliding_window_view(x_indx, window_shape=(k_size, k_size), axis=(2, 3))
    sw_indx = sw_indx[:, :, ::s, ::s, :, :]

    sw = sliding_window_view(X, window_shape=(k_size, k_size), axis=(2, 3))
    sw = sw[:, :, ::s, ::s, :, :]
    print("stride: ", s)
    print("sw shape: ", sw.shape)
    print("sw_indx shape: ", sw_indx.shape)
    #print("sw: ", sw)
    #print("sw_indx: ", sw_indx)

    total = np.prod(sw.shape)
    m = sw.reshape((int(total/(k_size**2)), k_size**2))
    m_indx = sw_indx.reshape((int(total/(k_size**2)), k_size**2))

    print("m shape: ", m.shape)
    print("m_indx shape: ", m_indx.shape)
    print("m_indx: ", m_indx)

    # receptive field argmax per row
    amax = np.argmax(m, axis=1)
    print("amax: ", amax)
    rows = np.arange(m.shape[0])
    amax_indx = m_indx[rows, amax]
    print("amax_indx: ", amax_indx)
    z = np.zeros(np.prod(X.shape))
    dout = np.random.random(np.prod(amax.shape))

    for i, d in enumerate(dout):
       z[amax_indx[i]] += d


    print("z: ", z.reshape(X.shape))


shape = (3, 3, 3, 4)
X = np.arange(0, np.prod(shape)*2, 2).reshape(shape)
k_size = 2
K = np.arange(k_size**2).reshape(k_size, k_size)
print("X: ", X)
#print("K: ", K)

run(X, K, 1)
