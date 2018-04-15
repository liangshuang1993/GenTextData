# -*- coding: utf-8 -*-
def is_chinese(word):
    for ch in word:
        # uc=ch.decode('utf-8')
        if u'\u4e00' <= ch <= u'\u9fff':
            continue
        else:
            return False
    return True


f_w = open('new_label.txt', 'w')


with open('labels.txt', 'r') as f:
    labels = f.readlines()
    for label in labels:
        label = label.strip().decode('utf-8')
        if is_chinese(label):
            label_list = list(set(label))
            label_list.append(label)
            for new_label in label_list:
                print new_label
                f_w.write(new_label.encode('utf-8'))
                f_w.write('\n')
        else:
            f_w.write(label.encode('utf-8'))
            f_w.write('\n')

f_w.close()
