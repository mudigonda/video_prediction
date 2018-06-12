import sys
sys.path.append('/home/ubuntu/Projects/video_prediction/')
import argparse
import glob
import itertools
import os
import random
import re
import numpy as np
from multiprocessing import Pool
import pickle

#import cv2
import tensorflow as tf

from video_prediction.datasets.base_dataset import VideoDataset


class GelSightVideoDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        super(GelSightVideoDataset, self).__init__(*args, **kwargs)
        self.state_like_names_and_shapes['images'] = '%d/images/encoded', (100, 100, 3)
        self.action_like_names_and_shapes['actions'] = '%d/actions', (2,)
        self._check_or_infer_shapes()

    def get_default_hparams_dict(self):
        default_hparams = super(GelSightVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=2,
            sequence_length=15,
            use_state=True,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    @property
    def jpeg_encoding(self):
        return False

    def num_examples_per_epoch(self):
        # extract information from filename to count the number of trajectories in the dataset
        count = 0
        for filename in self.filenames:
            match = re.search('sequence_(\d+)_to_(\d+).tfrecords', os.path.basename(filename))
            start_traj_iter = int(match.group(1))
            end_traj_iter = int(match.group(2))
            count += end_traj_iter - start_traj_iter + 1

        # alternatively, the dataset size can be determined like this, but it's very slow
        # count = sum(sum(1 for _ in tf.python_io.tf_record_iterator(filename)) for filename in filenames)
        return count


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def partition_data(input_dir):
    train_list_fnames = glob.glob(os.path.join(input_dir, '*.log'))
    fnames = [train_fname for train_fname in train_list_fnames]

    #random.shuffle(train_fnames)

    pivot1 = int(0.85 * len(fnames))
    pivot2 = int(0.95 * len(fnames))
    train_fnames, val_fnames, test_fnames = fnames[:pivot1], fnames[pivot1:pivot2], fnames[pivot2:]
    return train_fnames, val_fnames, test_fnames


def read_log(fname):
    if not os.path.isfile(fname):
        raise FileNotFoundError
    log_data = pickle.load(open(fname,'rb'))
    frames = log_data['states']
    actions = log_data['actions']
    return frames,actions


def save_tf_record(output_fname, sequences, actions):
    print('saving sequences to %s' % output_fname)
    feature = {}
    with tf.python_io.TFRecordWriter(output_fname) as writer:
        for ii, sequence in enumerate(sequences):
            num_frames = len(sequence)
            height, width, channels = sequence[0].shape
            #have added stochasticity to the data
            encoded_sequence = [tf.compat.as_bytes((np.array(image + np.random.randint(0,3))).tobytes()) for image in sequence]
            print("This sequence has num_frames {}".format(num_frames))
            print("This sequence has height and width are  {} {} {}".format(height, width, channels))
            print("This sequence has images of type {} ".format(type(encoded_sequence)))
            for  t, encoded_im in enumerate(encoded_sequence):
              feature['%d/actions' % t] = _float_feature(actions[t].tolist())
              feature['%d/images/encoded' % t] = _bytes_feature(encoded_im)
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def read_logs_and_save_tf_records(output_dir, fnames, start_sequence_iter=None,
                                    end_sequence_iter=None, sequences_per_file=128):
    print('started process with PID:', os.getpid())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if start_sequence_iter is None:
        start_sequence_iter = 0
    if end_sequence_iter is None:
        end_sequence_iter = len(fnames)


    print('reading and saving sequences {0} to {1}'.format(start_sequence_iter, end_sequence_iter))

    sequences = []
    actions = []
    for sequence_iter in range(start_sequence_iter, end_sequence_iter):
        if not sequences:
            last_start_sequence_iter = sequence_iter
            print("reading sequences starting at sequence %d" % sequence_iter)

        seq, act = read_log(fnames[sequence_iter])
        sequences.append(seq)
        actions.append(act)

        if len(sequences) == sequences_per_file or sequence_iter == (end_sequence_iter - 1):
            output_fname = 'sequence_{0}_to_{1}.tfrecords'.format(last_start_sequence_iter, sequence_iter)
            output_fname = os.path.join(output_dir, output_fname)
            save_tf_record(output_fname, sequences, act )
            sequences[:] = []


def main():
    parser = argparse.ArgumentParser()
    print("Initialized ArgumentParser")
    parser.add_argument("--input_dir", type=str, help="directory containing the directories of ")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument('--num_workers', type=int, default=1, help='number of parallel workers')
    args = parser.parse_args()

    partition_names = ['train', 'val', 'test']
    partition_fnames = partition_data(args.input_dir)

    for partition_name, partition_fnames in zip(partition_names, partition_fnames):
        partition_dir = os.path.join(args.output_dir, partition_name)
        print("Partition Name is {}".format(partition_name))
        if not os.path.exists(partition_dir):
            os.makedirs(partition_dir)

        if args.num_workers > 1:
            num_seqs_per_worker = len(partition_fnames) // args.num_workers
            start_seq_iters = [num_seqs_per_worker * i for i in range(args.num_workers)]
            end_seq_iters = [num_seqs_per_worker * (i + 1) - 1 for i in range(args.num_workers)]
            end_seq_iters[-1] = len(partition_fnames)

            p = Pool(args.num_workers)
            p.starmap(read_logs_and_save_tf_records, zip([partition_dir] * args.num_workers,
                                                           [partition_fnames] * args.num_workers,
                                                           start_seq_iters, end_seq_iters))
        else:
            read_logs_and_save_tf_records(partition_dir, partition_fnames)


if __name__ == '__main__':
    main()
