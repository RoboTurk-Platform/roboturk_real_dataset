'''
Code for demoing and iterating through the hdf5
'''

import h5py
import os
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import argparse


def demo_hdf5(f):
    data = f['data']

    for key in data.keys():

        user = data[key]
        demo_ids = user.keys()

        print('--group user: {}'.format(key))
        
        for demo_id in demo_ids:
            print('----group demo: {}'.format(demo_id))

            
            all_demo_attrs = dict(user[demo_id].attrs)
            for k in all_demo_attrs:
                attribute = user[demo_id].attrs[k]
                print('----demo attribute: {} with value: {}'.format(k, attribute))
            robot_obs_keys = user[demo_id]['robot_observation'].keys()
        
            print('-------group robot observation')
            for k in robot_obs_keys:
                obs_shape = user[demo_id]['robot_observation'][k].shape
                print('--------robot observation dataset {} with {} shape'.format(k, obs_shape))

            user_keys = user[demo_id]['user_control'].keys()

            print('-------group user control')
            for k in user_keys:
                user_shape = user[demo_id]['user_control'][k].shape
                print('------user control dataset {} with {} shape'.format(k, user_shape))
            break
        break

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--hdf5_input', required=True, help='HDF5 file to parse')

    results = parser.parse_args()

    f = h5py.File(results.hdf5_input, 'r')

    demo_hdf5(f)

if __name__ == "__main__":
    main()
