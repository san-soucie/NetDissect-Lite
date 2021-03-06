import torch
import torch.nn as nn
import torchvision
import os
import argparse

from matplotlib import pyplot as plt


class NetSoftmax(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):
        super(NetSoftmax, self).__init__()
        self.net = torchvision.models.alexnet(
            pretrained=True, num_classes=num_classes)
        #                                            aux_logits=aux_logits,
        #                                            transform_input=transform_input)
        #        self.net.classifier = nn.Linear(self.net.classifier.in_features*81, num_classes)
        self.sm = nn.Softmax()

    def forward(self, x):
        x = self.net(x)
        x = self.sm(x)
        return x


def get_params(filename):
    v = os.path.split(filename.rstrip('__0.png'))[1].split('_')
    unit, category, label, score, layer = v[0:5]
    hp = v[5:]
    params = {'unit': unit,
              'category': category,
              'label': label,
              'score': score,
              'layer': layer,
              'hyperparameters': hp}
    return params


def process_target_class(filename):
    return get_params(filename)['label']


def process_score(filename):
    return get_params(filename)['score']


def find_classes(class_filename):
    with open(class_filename) as f:
        classes = [x.strip() for x in f.readlines()]
    class_to_idx = {y: i for i, x in enumerate(classes) for y in x.split(', ')}
    full_class_to_idx = {y: i for i, y in enumerate(classes)}
    return classes, {**class_to_idx, **full_class_to_idx}


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self,
                 folder_name,
                 transform=None,
                 class_filename="imagenet_classes.txt",
                 stopwords=None):

        classes, class_to_idx = find_classes(class_filename)

        self.filenames = [os.path.join(folder_name, x) for x in os.listdir(folder_name)]
        removes = set()
        for stopword in stopwords:
            for f in self.filenames:
                if stopword in f:
                    removes |= {f}
        self.filenames = [f for f in self.filenames if f not in removes]
        self.root = folder_name
        samples = [filename for filename in self.filenames]
        scores = {filename: float(process_score(filename))
                  for filename in self.filenames}

        self.loader = torchvision.datasets.folder.pil_loader
        self.extensions = torchvision.datasets.folder.IMG_EXTENSIONS
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.transform = transform
        self.scores = scores

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        return self.transform(sample) if self.transform else sample, self.get_score(index), path

    def get_score(self, index):
        path = self.samples[index]
        return self.scores[path]

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def main():
    parser = argparse.ArgumentParser(description="Creates a scatterplot of scores vs. softmax probabilities")
    parser.add_argument("image_folder_name", default="/home/warp/Documents/6.861/NetDissect-Lite/outputs/synthesizing-06:52:34/CORnet-S/clean",
                        metavar="F", help="Folder containing images to be parsed")
    parser.add_argument("-T", "--network_name",
                        metavar="T", default="CORnet-S", help="Name of neural network generating synthesized images")
    parser.add_argument("-O", "--output_image_filename",
                        metavar="F", default="CORnet-S_Clean_MaxSofmax_vs_IoU.pdf", help="Filename for output image")
    args = parser.parse_args()
    args.image_folder_names = []
    if args.image_folder_name == "LIST":
        args.image_folder_names = ["/home/warp/Documents/6.861/NetDissect-Lite/processing/example_synthesized_images"
                                   "/synthesizing-02:07:37",
                                   "/home/warp/Documents/6.861/NetDissect-Lite/processing/example_synthesized_images"
                                   "/synthesizing-02:20:25",
                                   "/home/warp/Documents/6.861/NetDissect-Lite/processing/example_synthesized_images"
                                   "/synthesizing-02:24:08",
                                   "/home/warp/Documents/6.861/NetDissect-Lite/processing/example_synthesized_images"
                                   "/synthesizing-02:25:48",
                                   "/home/warp/Documents/6.861/NetDissect-Lite/processing/example_synthesized_images"
                                   "/synthesizing-04:25:46"]
    if not args.image_folder_names:
        dataset = CustomDataset(args.image_folder_name,
                                transform=torchvision.transforms.ToTensor(),
                                stopwords={})
        dataloader = torch.utils.data.DataLoader(dataset)
        net = NetSoftmax()
        softmax_scores_object = []
        iou_scores_object = []
        softmax_scores_bad = []
        iou_scores_bad = []
        softmax_scores_texture = []
        iou_scores_texture = []
        with torch.no_grad():
            for image, score, path in dataloader:
                softmaxes = net.forward(image)
                if '_bad.' in path[0]:
                    softmax_scores_bad.append(max(softmaxes.tolist()[0]))
                    iou_scores_bad.append(score[0].item())
                elif "_object_" in path[0]:
                    softmax_scores_object.append(max(softmaxes.tolist()[0]))
                    iou_scores_object.append(score[0].item())
                else:
                    softmax_scores_texture.append(max(softmaxes.tolist()[0]))
                    iou_scores_texture.append(score[0].item())
    else:
        datasets = [CustomDataset(x,
                                  transform=torchvision.transforms.ToTensor(),
                                  stopwords={'CORnet'}) for x in args.image_folder_names]
        dataloaders = [torch.utils.data.DataLoader(d) for d in datasets]
        net = NetSoftmax()
        softmax_scores_object = []
        iou_scores_object = []
        softmax_scores_bad = []
        iou_scores_bad = []
        softmax_scores_texture= []
        iou_scores_texture = []
        with torch.no_grad():
            for d in dataloaders:
                for image, score, path in d:
                    softmaxes = net.forward(image)
                    if '_bad.' in path[0]:
                        softmax_scores_bad.append(max(softmaxes.tolist()[0]))
                        iou_scores_bad.append(score[0].item())
                    elif "_object_" in path[0]:
                        softmax_scores_object.append(max(softmaxes.tolist()[0]))
                        iou_scores_object.append(score[0].item())
                    else:
                        softmax_scores_texture.append(max(softmaxes.tolist()[0]))
                        iou_scores_texture.append(score[0].item())
    plt.plot(iou_scores_object, softmax_scores_object, 'bo', label="Top-5 Unit (Object)")
    plt.plot(iou_scores_texture, softmax_scores_texture, 'go', label="Top-5 Unit (Texture)")
    plt.plot(iou_scores_bad, softmax_scores_bad, 'ro', label="Bottom-5 Unit")

    if args.network_name:
        plt.title("Maximum softmax probability vs. IoU score (" + args.network_name + ")")
    else:
        plt.title("Maximum softmax probability vs. IoU score")
    plt.xlim(left=0, right=0.6)
    plt.ylim(bottom=0, top=1)
    plt.xlabel('Unit IoU Score')
    plt.ylabel('Maximum Softmax Score')
    plt.legend()
    img_filename = (args.output_image_filename
                    if args.output_image_filename
                    else args.network_name + "softmax_vs_iou.pdf")
    plt.savefig(img_filename, bbox_inches='tight')



if __name__ == "__main__":
    main()

