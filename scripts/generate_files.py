'''
Code for reading the aligned hdf5 and correlating the demos
to the appropriate USB camera video for video prediction
'''

import h5py
import os
import pickle
from collections import defaultdict
import numpy as np
import argparse


def generate_dataset(hdf5_file, output, video_dir):
    
    data = f['data']

    for key in data.keys():

        user = data[key]
        demo_ids = user.keys()

        for demo_id in demo_ids:
            
            usb_video_filepath = dict(user[demo_id].attrs)['front_rgb_video_file']
            full_usb_video = os.path.join(video_dir, usb_video_filepath)
            unique_demo = '{}_{},{}\n'.format(key, demo_id, full_usb_video)
            output.write(unique_demo)

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--hdf5_input', required=True, help='HDF5 file to parse')
    parser.add_argument('--output', required=True, help='Output dictionary to map demo to USB Camera Video')
    parser.add_argument('--video_dir', required=True, help='Prepends location of video directory to output')

    results = parser.parse_args()

    f = h5py.File(results.hdf5_input, 'r')

    output = open(results.output, "w")

    generate_dataset(f, output, video_dir)


if __name__ == "__main__":
    main()
