import numpy as np
import random

def test_validation():
    total_rows = 10
    total_columns = 8

    def get_data(total_rows, total_features):
        l = []
        s = 0
        for i in range(total_rows):
            l.append([i for i in range(s, s+total_features)])
            s += total_features
        return l

    X = get_data(total_rows, total_columns)
    y = get_data(total_rows, 1)
    print("X: ", X)
    print("y: ", y)

    def split(c, data, label):
        n1 = round(c*len(data))
        n2 = len(data) - n1
        train_data = data[:n1]
        train_label = label[:n1]
        val_data = data[-n2:]
        val_label = label[-n2:]
        return train_data, train_label, val_data, val_label


    td, tl, vd, vl = split(0.8, X, y)

    print("training X: ", td)
    print("training y: ", tl)
    print("val X: ", vd)
    print("val y: ", vl)

    def generate_batch(x, batch_size, shuffle = False):
        if shuffle:
            random.shuffle(x)

        l_batch = []
        s = 0
        while (s < len(x) - batch_size):
          nx = np.array(x[s:s+batch_size])
          l_batch.append(nx)
          s += batch_size

        nx = np.array(x[s:])
        l_batch.append(nx)

        return l_batch

    print("batch, size 3, training X: ", generate_batch(td, 3, True))
    print("batch, size 3, training y: ", generate_batch(tl, 3, True))


def test_accuracy():
    x = np.array([[0.2, 0.5, 0.3], [0.5, 0.1, 0.4], [0.3, 0.3, 0.4]])
    y = np.array([1, 2, 0])

    amax = np.argmax(x, axis=1)
    match_total = amax[amax == y].shape[0]
    accuracy = match_total/y.shape[0]
    print("accuracy: ", accuracy)

#test_validation()
test_accuracy()

