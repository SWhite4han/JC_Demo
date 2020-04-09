import os
import numpy as np
import Levenshtein

if __name__ == "__main__":
    label_path = os.path.abspath(os.path.join(__file__, "..", "result", "recognition", "word.xml"))
    result_path = os.path.abspath(os.path.join(__file__, "..", "result", "recognition", "result.xml"))

    with open(label_path, 'r', encoding="iso-8859-1") as fr:
        raw_labels = fr.readlines()
    with open(result_path, 'r') as fr:
        raw_preds = fr.readlines()

    labels_dict = dict()
    results_dict = dict()

    for idx, line in enumerate(raw_labels):
        if "image file" in line:
            labels_dict[line[line.index("word/"):line.index(".jpg")]] = line[line.index("tag=")+len('tag="'):line.index('" />')]

    for idx, line in enumerate(raw_preds):
        if "image file" in line:
            results_dict[line[line.index("word/"):line.index(".jpg")]] = line[line.index("tag=")+len('tag="'):line.index('"/>')]

    corrects = 0
    distances = 0
    total = 0
    ratios_list = list()
    for key in set(labels_dict).union(set(results_dict)):
        total += 1
        if labels_dict[key] == results_dict[key]:
            corrects += 1

        distances += Levenshtein.distance(labels_dict[key], results_dict[key])
        ratio = Levenshtein.ratio(labels_dict[key], results_dict[key])
        ratios_list.append(ratio)

    ratios = np.mean(np.asarray(ratios_list))

    print("accuracy: %.4f" % (corrects/total))
    print("mean similarity: %.4f" % ratios)
