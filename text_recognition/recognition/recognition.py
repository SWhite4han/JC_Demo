import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from text_recognition.recognition.utils import CTCLabelConverter, AttnLabelConverter
from text_recognition.recognition.dataset import RawDataset, AlignCollate
from text_recognition.recognition.model import Model


import xml.dom.minidom
def write_xml(results, img_paths):
    doc = xml.dom.minidom.Document()
    doc.toprettyxml(encoding="iso-8859-1")
    first_child = doc.createElement("imagelist")
    doc.appendChild(first_child)
    for result, img_path in zip(results, img_paths):
        # create a new XML tag and add it into the document
        newexpertise = doc.createElement("image")
        newexpertise.setAttribute("file", img_path)
        newexpertise.setAttribute("tag", result)
        first_child.appendChild(newexpertise)

    f = open('../result/result.xml', 'w')
    doc.writexml(f, indent="  ", addindent="  ", newl='\n')

def data_load(args):
    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=args.imgH, imgW=args.imgW, keep_ratio_with_pad=args.PAD)
    demo_data = RawDataset(root=args.image_folder, opt=args)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)
    return demo_loader


def demo(args):
    model = Recognition(args)

    # predict
    demo_loader = data_load(args)
    results = []
    img_paths = []
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:

            preds_str = model.predict(image_tensors)
            # print('-' * 80)
            # print('image_path\tpredicted_labels')
            # print('-' * 80)
            for img_name, pred in zip(image_path_list, preds_str):
                if 'Attn' in args.Prediction:
                    pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])

                # print(f'{img_name}\t{pred}')
                results.append(pred)
                img_paths.append(img_name)
    # write_xml(results, img_paths)
    return results, img_paths


class Recognition:
    def __init__(self, args):
        """ model configuration """
        self.args = args
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if 'CTC' in self.args.Prediction:
            self.converter = CTCLabelConverter(self.args.character)
        else:
            self.converter = AttnLabelConverter(self.args.character)
        self.args.num_class = len(self.converter.character)

        if self.args.rgb:
            self.args.input_channel = 3
        self.model = Model(self.args)
        self.model = torch.nn.DataParallel(self.model).to(device)

        # load model
        print('loading pretrained model from %s' % self.args.recog_model)
        self.model.load_state_dict(torch.load(self.args.recog_model))
        self.model.eval()

    def predict(self, image):
        batch_size = image.size(0)
        if batch_size==0:
            preds_str = []
            return preds_str
        image = image.cuda()
        # For max length prediction
        length_for_pred = torch.IntTensor([self.args.batch_max_length] * batch_size).cuda()
        text_for_pred = torch.LongTensor(batch_size, self.args.batch_max_length + 1).fill_(0).cuda()

        if 'CTC' in self.args.Prediction:
            preds = self.model(image, text_for_pred).log_softmax(2)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.permute(1, 0, 2).max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = self.converter.decode(preds_index.data, preds_size.data)

        else:
            preds = self.model(image, text_for_pred)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)
        return preds_str


def recog_params():
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--recog_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                       help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = recog_params()
    # recognition arg
    # parser = argparse.ArgumentParser()
    # recog_group = parser.add_argument_group(title="recog")
    # recog_params(recog_group)


    """ vocab / character number configuration """
    if args.sensitive:
        args.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    cudnn.benchmark = True
    cudnn.deterministic = True
    args.num_gpu = torch.cuda.device_count()

    # demo(args)
