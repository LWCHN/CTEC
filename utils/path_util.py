import os
import re

def makeDIR(*path):
    for new_path in path:
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        assert(os.path.exists(new_path))
    return new_path



def dir_reader(data_dir, filter_pattern=None, sub_dir=None, depth=100):
    assert isinstance(depth, int)
    depth = max(depth - 1, 0)
    full_paths = []
    cur_paths = []
    if depth:
        current_dir = data_dir if not sub_dir else os.path.join(data_dir, sub_dir)
        sub_files = os.listdir(current_dir)
        for sub_file in sub_files:
            sub_file_path = sub_file if not sub_dir else os.path.join(sub_dir, sub_file)
            file_path = os.path.join(current_dir, sub_file)
            
            if os.path.isdir(file_path):
                sub_full_paths, sub_cur_paths = dir_reader(data_dir, filter_pattern, sub_dir=sub_file_path, depth=depth)
                full_paths += sub_full_paths
                cur_paths += sub_cur_paths
            elif not filter_pattern or re.match(filter_pattern, sub_file_path):
                full_paths.append(os.path.join(data_dir, sub_file_path))
                cur_paths.append(sub_file_path)
            else:
                pass
                #print('filter', file_path)
    return full_paths, cur_paths

def win_to_linux(win_path):
    return win_path.replace('\\','/')