import os
import tensorflow as tf
import xml.etree.ElementTree as ET
from object_detection.utils import dataset_util

# Change this to your label map
LABEL_DICT = {
    'acne': 1,
    'oily': 2,
    'dry': 3,
    'wrinkle': 4,
    'darkspot': 5
}

def create_tf_example(xml_path, image_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename').text
    image_path = os.path.join(image_dir, filename)

    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()

    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        xml_box = obj.find('bndbox')

        xmins.append(float(xml_box.find('xmin').text) / width)
        xmaxs.append(float(xml_box.find('xmax').text) / width)
        ymins.append(float(xml_box.find('ymin').text) / height)
        ymaxs.append(float(xml_box.find('ymax').text) / height)
        classes_text.append(label.encode('utf8'))
        classes.append(LABEL_DICT[label])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(b'jpg'),  # or 'png'
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_dir', help='Path to folder containing XML files')
    parser.add_argument('--image_dir', help='Path to folder containing images')
    parser.add_argument('--output_path', help='Path to output TFRecord')
    args = parser.parse_args()

    writer = tf.io.TFRecordWriter(args.output_path)
    xml_files = [f for f in os.listdir(args.xml_dir) if f.endswith('.xml')]

    for xml_file in xml_files:
        xml_path = os.path.join(args.xml_dir, xml_file)
        tf_example = create_tf_example(xml_path, args.image_dir)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print(f'Successfully created TFRecord at {args.output_path}')

if __name__ == '__main__':
    tf.compat.v1.app.run()
