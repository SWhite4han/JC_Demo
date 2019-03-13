import tempfile
from os import listdir, makedirs
from os.path import isdir, join
import shutil
import json

# named_gen = tempfile.NamedTemporaryFile(dir='')
# print([tempfile.NamedTemporaryFile(dir='', prefix='PER_').name for i in range(5)])


def rename_dirs(path, out_path):
    if isdir(out_path):
        shutil.rmtree(out_path)
    dirs = [f for f in listdir(path) if isdir(join(path, f))]
    info = {'id_to_name': dict(),
            'name_to_id': dict()}
    for dir_name in dirs:
        temp = tempfile.NamedTemporaryFile(prefix='PER_')
        temp_id = temp.name.split('/')[-1]
        info['id_to_name'][temp_id] = dir_name
        info['name_to_id'][dir_name] = temp_id
        shutil.copytree(join(path, dir_name), join(out_path, temp.name.split('/')[-1]))
    with open(join(out_path, 'info.json'), 'w') as wf:
        json.dump(info, wf, ensure_ascii=False)


if __name__ == '__main__':
    rename_dirs('/data1/images/0821/manaulLabel/known_people_manually', '/data1/images/0821/manaulLabel/known_people_id')

