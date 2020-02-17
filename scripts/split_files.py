'''
Code for splitting the demos into train/test/split for video prediction

Usage:
python split_files \
    --files=all_demos.txt \
    --train_split=0.7 \
    --eval_split=0.2 \
    --test_split=0.1 \
    --problem=SawyerLaundryLayout
'''
import argparse
import os
import random

def write_demos_to_file(problem, files, split_name):
    with open('{}_{}.txt'.format(problem, split_name), 'w') as file_obj:
        for f in files:
            file_obj.write("{}\n".format(f))

def split_files(filepath, problem, train_split, eval_split, test_split):
    '''
    split_files splits the files into train/eval/test according to the splits
    '''
    all_files = []
    with open(filepath, 'r') as f:
        for cnt, line in enumerate(f):
            line = line.rstrip()
            all_files.append(line)
    random.shuffle(all_files)
    train_idx = int(len(all_files) * train_split)
    eval_idx = int(train_idx + len(all_files) * eval_split)
    test_idx = int(train_idx + eval_idx + len(all_files) * test_split)

    train_files = all_files[:train_idx]
    eval_files = all_files[train_idx:eval_idx]
    test_files = all_files[eval_idx:]

    write_demos_to_file(problem, train_files, 'train')
    write_demos_to_file(problem, eval_files, 'eval')
    write_demos_to_file(problem, test_files, 'test')

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--files', help='Files to split')
    parser.add_argument('--train_split', type=float, help='Train Split')
    parser.add_argument('--eval_split', type=float, help='Eval Split')
    parser.add_argument('--test_split', type=float, help='Test Split')
    parser.add_argument('--problem', type=float, help='Identifying problem')

    results = parser.parse_args()

    if(results.train_split + results.eval_split + results.test_split != 1.0):
        raise ValueError("Incorrect split fractions")

    split_files(results.files, results.problem, results.train_split, results.eval_split, results_test_split) 


if __name__ == "__main__":
    main()
