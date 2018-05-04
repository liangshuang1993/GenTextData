import os
import numpy as np


# get dicts of train and val
def get_dict(train_label, val_label):
    dicts = []

    with open(train_label, 'r') as f:
        train_labels = f.readlines()

    with open(val_label, 'r') as f:
        val_labels = f.readlines()

    for label in train_labels:
        label = label.strip().decode('utf-8').split(' ')[1]
        for char in label:
            dicts.append(char)
    for label in val_labels:
        label = label.strip().decode('utf-8').split(' ')[1]
        for char in label:
            dicts.append(char)

    dicts = list(set(dicts))

    with open('dicts.txt', 'w') as f:
        for char in dicts:
            f.write(char.encode('utf-8'))


def increase_label(original, target):
    with open(original, 'r') as f:
        lines = f.readlines()


    with open(target, 'w') as f:
        for line in lines:
            line = line.strip().decode('utf-8')
            f.write(line.encode('utf-8') + '\n')

            # random
            length = len(line)
            if length > 1:
                for i in range(length):
                    start = np.random.randint(length)
                    end = np.random.randint(start, length)

                    if end - start > 6:
                        continue

                    label = line[start: end + 1]
                    f.write(label.encode('utf-8') + '\n')

    with open('dicts.txt', 'r') as f:
        dicts = f.readlines()[0].strip().decode('utf-8')
        dicts = list(dicts)

    with open(target, 'a+') as f:
        for i in range(10000):
            # choose random word
            # number = np.random.randint(2, 7)
            number = 5
            words = np.random.choice(dicts, number)
            # print words.encode('utf-8')
            # print words
            line = ''.join(words)
            f.write(line.encode('utf-8') + '\n')

if __name__ == '__main__':
    # get_dict('datasets_c/train.txt', 'val.txt')
    increase_label('empty.txt', 'new_labels_fix.txt')
