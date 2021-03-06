import numpy as np
import tensorflow as tf

class Dataset:
    def csv_reader_dataset(filepaths, preprocess=None, repeat=20, n_readers=5,
                        n_read_threads=None, shuffle_buffer_size=10000,
                        n_parse_threads=5, batch_size=32):
        dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
        dataset = dataset.interleave(
            lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
            cycle_length=n_readers, num_parallel_calls=n_read_threads)
        dataset = dataset.shuffle(shuffle_buffer_size)
        if preprocess:
            dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(1)
    
    def csv_to_np_dataset(filepath):
        return np.loadtxt(filepath, delimiter=',', skiprows=1)

if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(os.getcwd())
    filepath = '../../data/train.csv'
    dataset = Dataset.csv_to_np_dataset(filepath)
    print(dataset[:5])