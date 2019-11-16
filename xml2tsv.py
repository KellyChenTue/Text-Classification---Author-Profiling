"""
xml2tsv.py: convert xml file to tsv file

Author: Chen, Pin-Zhen

Prepare dataset for author_profiling training. Get the gender and the text.
Please adapt to your usage. e.g : changing the path to the files.
"""

import os
import xml.etree.ElementTree as ET
import csv


def xml2tsv(filename):
    pathname = '/Users/chenpinzhen/snlp2019/external_data/data/' + filename
    tree = ET.parse(pathname)

    root = tree.getroot()
    # get gender of the author:
    # root.attrib['gender']  , return female or male
    gender = root.attrib['gender']


    # for external file, I got xml file from PAN dataset
    # xml2tsv method is to get gender and text
    with open('data.tsv', 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        for child in root:
            for sib in child:
                    if gender == 'male':
                        gen = 'm'
                    else:
                        gen = 'f'
                    writer.writerow([gen , sib.text.strip()])


if __name__ == '__main__':
    with open('data.tsv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Label', 'Text'])
    for root, dirnames, filenames in os.walk(
            '/Users/chenpinzhen/snlp2019/external_data/data'):
        for filename in filenames:
            if filename.endswith('.xml'):
                xml2tsv(filename)
