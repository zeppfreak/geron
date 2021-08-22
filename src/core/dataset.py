import tensorflow as tf

class Dataset:
    def csv_reader_dataset(filepaths, preprocess=None, repeat=1, n_readers=5,
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

@tf.function
def preprocess(line):
    defs = [0.] * 8 + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return x,y

if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(os.getcwd())
    filepaths = ['../../data/train.csv']
    dataset = Dataset.csv_reader_dataset(filepaths)