import os
import numpy as np
from pandas import DataFrame

SKIP_FILES = {'cmds'}

def load_data_files(path):
    data = DataFrame({'text': [], 'class': []})
    classes = list()
    for folder in os.listdir(path):
        p = os.path.join(path, folder)
        classes.append(folder)
        data = data.append(build_data_frame(p, folder))
    data = data.reindex(np.random.permutation(data.index))
    print classes

def build_data_frame(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'class': classification})
        index.append(file_name)
    data_frame = DataFrame(rows, index=index)
    return data_frame

def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        print "root: ", root
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            print "file name: ", file_name
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    f = open(file_path)
                    text = f.read()
                    f.close()
                    yield file_path, text
