import os
import xml.etree.ElementTree as ET


class icdar2003(object):
    def __init__(self, OCD_dataset_path, OCR_dataset_path):
        """
        compute accuracy by region of union
        dataset: icdar 2003
        """

        self.OCD_dataset_path = OCD_dataset_path
        self.OCR_dataset_path = OCR_dataset_path

        self.OCD_data_paths = list()
        self.OCR_data_paths = list()
        self.end_to_end_data_paths = list()

        self.OCD_labels = list()
        self.OCR_labels = list()
        self.end_to_end_labels = list()

        self.end_to_end_box_positions = list()

    def OCD_dataset(self):
        tree = ET.parse(os.path.join(self.OCD_dataset_path, "locations.xml"))
        root = tree.getroot()

        for image_info in root:
            for element in image_info:
                if element.tag == "imageName":
                    self.OCD_data_paths.append(os.path.join(self.OCD_dataset_path, element.text))

                if element.tag == "taggedRectangles":
                    boxes_list = list()
                    for boxes in element:
                        box = [
                            [float(boxes.attrib['x']), float(boxes.attrib['y'])],  # upper left
                            [float(boxes.attrib['x']) + float(boxes.attrib['width']), float(boxes.attrib['y'])],
                            # upper right
                            [float(boxes.attrib['x']) + float(boxes.attrib['width']),
                             float(boxes.attrib['y']) + float(boxes.attrib['height'])],  # bottom right
                            [float(boxes.attrib['x']), float(boxes.attrib['y']) + float(boxes.attrib['height'])]
                            # bottom left
                        ]
                        boxes_list.append(box)

                    self.OCD_labels.append(boxes_list)

        return self.OCD_data_paths, self.OCD_labels

    def OCR_dataset(self):
        tree = ET.parse(os.path.join(self.OCR_dataset_path, "word.xml"))
        root = tree.getroot()

        for child in root:
            self.OCR_data_paths.append(os.path.join(self.OCR_dataset_path, child.attrib['file']))
            self.OCR_labels.append(child.attrib['tag'])

        return self.OCR_data_paths, self.OCR_labels

    def end_to_end_dataset(self):
        """
        end_to_end = OCD + OCR
        using scene images as input, but characters as output
        :return:
        """
        tree = ET.parse(os.path.join(self.OCD_dataset_path, "words.xml"))
        root = tree.getroot()

        for image_info in root:
            for element in image_info:
                if element.tag == "imageName":
                    self.end_to_end_data_paths.append(os.path.join(self.OCD_dataset_path, element.text))

                if element.tag == "taggedRectangles":
                    boxes_list = list()
                    text_list = list()
                    for box in element:  # element including many boxes
                        # get text
                        for info in box:
                            if info.tag == "tag":
                                text_list.append(info.text)

                        # get position
                        box_info = [
                            [float(box.attrib['x']), float(box.attrib['y'])],  # upper left
                            [float(box.attrib['x']) + float(box.attrib['width']), float(box.attrib['y'])],  # upper right
                            [float(box.attrib['x']) + float(box.attrib['width']),
                             float(box.attrib['y']) + float(box.attrib['height'])],  # bottom right
                            [float(box.attrib['x']), float(box.attrib['y']) + float(box.attrib['height'])],  # bottom left
                        ]
                        boxes_list.append(box_info)

                    if len(text_list) != len(boxes_list):
                        print("data broken: %s" % (self.end_to_end_data_paths.pop()))
                        continue

                    self.end_to_end_labels.append(text_list)
                    self.end_to_end_box_positions.append(boxes_list)

        # check
        if len(self.end_to_end_data_paths) != len(self.end_to_end_labels) and len(self.end_to_end_data_paths) != len(self.end_to_end_box_positions):
            print("dataset borken, number of data, label and boxes aren't the same:(%d, %d, %d)" % (
                len(self.end_to_end_data_paths), len(self.end_to_end_labels), len(self.end_to_end_box_positions)))
            exit()

        return self.end_to_end_data_paths, self.end_to_end_labels, self.end_to_end_box_positions


# if __name__ == "__main__":
#     dataset = icdar2003(OCD_dataset_path="/data1/Dataset/OCR/icdar2003/Robust Reading and Text Locating/SceneTrialTest",
#                         OCR_dataset_path="/data1/Dataset/OCR/icdar2003/Robust Word Recognition/1")
#     data_paths, labels, box_positions = dataset.end_to_end_dataset()
#     pass
