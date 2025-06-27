# Script to create TFRecord files from train and test dataset folders
# Originally from GitHub user datitran: https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py

"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""
import os
import io
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple

# Define flags
flags = tf.app.flags
flags.DEFINE_string('csv_input', 'dataset_object_detection/annotations_labels.csv', 'Path to the CSV input')
flags.DEFINE_string('labelmap', 'dataset_object_detection/labelmap.txt', 'Path to the labelmap text file (1 class per line)')
flags.DEFINE_string('image_dir', 'dataset_object_detection/images', 'Path to the image directory')
flags.DEFINE_string('output_path', 'dataset_object_detection/train.record', 'Path to output TFRecord')
FLAGS = flags.FLAGS

# Helper function to split grouped dataframe
def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    grouped = df.groupby(group)
    return [data(filename, grouped.get_group(x)) for filename, x in zip(grouped.groups.keys(), grouped.groups)]

# Main conversion function
def create_tf_example(group, path, label_list):
    with tf.gfile.GFile(os.path.join(path, group.filename), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'  # or 'jpeg' or 'png'
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    for _, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        label = row['class']
        classes_text.append(label.encode('utf8'))
        classes.append(label_list.index(label) + 1)  # +1 because labelmap starts from 1

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

# Main function
def main(_):
    # Load labels
    with open(FLAGS.labelmap, 'r') as f:
        label_list = [line.strip() for line in f if line.strip()]

    # Write labelmap.pbtxt
    with open('labelmap.pbtxt', 'w') as f:
        for i, label in enumerate(label_list):
            f.write('item {\n')
            f.write(f'  id: {i+1}\n')
            f.write(f'  name: \'{label}\'\n')
            f.write('}\n\n')

    # Load CSV and create TFRecord
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')

    for group in grouped:
        tf_example = create_tf_example(group, FLAGS.image_dir, label_list)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print(f"✅ TFRecord created at: {FLAGS.output_path}")
    print(f"✅ Labelmap saved at: labelmap.pbtxt")

if __name__ == '__main__':
    tf.app.run()
