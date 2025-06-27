# Script to create CSV data file from Pascal VOC annotation files
# Based off code from GitHub user datitran: https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(xml_folder):
    xml_list = []
    for xml_file in glob.glob(os.path.join(xml_folder, '*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (
                root.find('filename').text,
                int(root.find('size')[0].text),   # width
                int(root.find('size')[1].text),   # height
                member.find('name').text,
                int(member.find('bndbox')[0].text),  # xmin
                int(member.find('bndbox')[1].text),  # ymin
                int(member.find('bndbox')[2].text),  # xmax
                int(member.find('bndbox')[3].text)   # ymax
            )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main():
    annotations_path = 'dataset_object_detection/annotations'
    output_csv_path = 'dataset_object_detection/annotations_labels.csv'

    xml_df = xml_to_csv(annotations_path)
    xml_df.to_csv(output_csv_path, index=False)
    print(f"✅ Successfully converted XML annotations to CSV: {output_csv_path}")

main()
