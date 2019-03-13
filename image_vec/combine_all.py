import os
import json
import jieba
import jieba.analyse
import numpy as np


def get_text(data_path):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['json']
    for parent, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} json files'.format(len(files)))
    return files


def load_detect():
    source_p = 'detect_source.npy'
    keword_p = 'detect_keywords.npy'
    source = np.load(os.path.join(root_path, source_p))
    keword = np.load(os.path.join(root_path, keword_p))
    return source, keword


def find_key(target, sources, values):
    for i in range(len(sources)):
        if target in sources[i]:
            return values[i]


def load_face():
    source_p = 'face_source.npy'
    vec_p = 'face_vectors.npy'
    source = np.load(os.path.join(root_path, source_p))
    vec = np.load(os.path.join(root_path, vec_p))
    return source, vec


def load_img_vec():
    source_p = 'img_source.npy'
    source_p1 = 'img_source1.npy'
    source_p2 = 'img_source2.npy'
    source_p3 = 'img_source3.npy'
    source_all = np.load(os.path.join(root_path, source_p)).tolist()
    source1 = np.load(os.path.join(root_path, source_p1)).tolist()
    source2 = np.load(os.path.join(root_path, source_p2)).tolist()
    source3 = np.load(os.path.join(root_path, source_p3)).tolist()

    source_all.extend(source1)
    source_all.extend(source2)
    source_all.extend(source3)

    vec_p = 'img_vectors.npy'
    vec_p1 = 'img_vectors1.npy'
    vec_p2 = 'img_vectors2.npy'
    vec_p3 = 'img_vectors3.npy'
    vec2 = np.load(os.path.join(root_path, vec_p2)).tolist()
    vec1 = np.load(os.path.join(root_path, vec_p1)).tolist()
    vec3 = np.load(os.path.join(root_path, vec_p3)).tolist()
    vec_all = np.load(os.path.join(root_path, vec_p)).tolist()

    vec_all.extend(vec2)
    vec_all.extend(vec1)
    vec_all.extend(vec3)
    return source_all, vec_all


if __name__ == '__main__':
    file_paths = get_text('/data1/JC_Sample/sample_data_news/text')
    root_path = '/data1/JC_Sample/sample_data_news/results'
    save_path = '/data1/JC_Sample/sample_data_news'

    # Read data QQ
    face_sources, face_vecs = load_face()
    img_sources, img_vecs = load_img_vec()
    detect_sources, detect_keywords = load_detect()

    for path in file_paths:
        with open(path, 'r') as f:
            news = json.load(f)

        keywords = jieba.analyse.textrank(news['Title'] + '\n' + news['Content'], topK=10, withWeight=False)
        if news['ImgPath']:
            for full_path in news['ImgPath']:
                new_dict = dict()
                relative_path = full_path.replace('/data2/Dslab_News/img/Chinatimes/', 'img/')
                new_dict['ImgPath'] = relative_path

                file_name = relative_path.split('/')[-1]
                file_name = file_name.replace('.jpg', '')

                kw = find_key(file_name, detect_sources, detect_keywords)
                if kw:
                    # combine redundancy objs
                    # Maybe count objs is great than this
                    keywords.extend(list(set(kw)))
                new_dict['KeyWord'] = keywords

                fv = find_key(file_name, face_sources, face_vecs)
                if fv is not None:
                    new_dict['FaceVec'] = fv.tolist()
                else:
                    new_dict['FaceVec'] = []

                iv = find_key(file_name, img_sources, img_vecs)
                if iv:
                    new_dict['ImgVec'] = iv
                else:
                    new_dict['ImgVec'] = []

                with open(os.path.join(save_path, file_name + '.json'), 'w') as wf:
                    json.dump(new_dict, wf, ensure_ascii=False)



