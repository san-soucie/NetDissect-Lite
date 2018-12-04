import torch
import torch.nn as nn
import torchvision
import os
import argparse
from matplotlib import pyplot as plt


class Inception3Softmax(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):
        super(Inception3Softmax, self).__init__()
        self.Inception3 = torchvision.models.inception_v3(pretrained=True,
                                                          num_classes=num_classes,
                                                          aux_logits=aux_logits,
                                                          transform_input=transform_input)
        self.sm = nn.Softmax()

    def forward(self, x):
        x = self.Inception3(x)
        x = self.sm(x)
        return x


def process_target_class(filename):
    pass


def process_score(filename):
    pass


def find_classes(class_filename):
    with open(class_filename) as f:
        classes = [x.strip() for x in f.readlines()]
    class_to_idx = {y: i for i, x in enumerate(classes) for y in x.split(', ')}
    full_class_to_idx = {y: i for i, y in enumerate(classes)}
    return classes, {**class_to_idx, **full_class_to_idx}


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, folder_name, transform=None, class_filename="imagenet_classes.txt"):

        classes, class_to_idx = find_classes(class_filename)

        self.filenames = os.listdir(folder_name)
        self.root = folder_name
        samples = [(filename, process_target_class(filename, class_to_idx)) for filename in self.filenames]
        scores = {filename: process_score(filename) for filename in self.filenames}
        self.loader = torchvision.datasets.folder.default_loader()
        self.extensions = torchvision.datasets.folder.IMG_EXTENSIONS
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.transform = transform
        self.scores = scores

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        return self.transform(sample) if self.transform else sample, target, self.get_score(index)

    def get_score(self, index):
        path, _ = self.samples[index]
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
    parser.add_argument("image_folder_name", metavar="F", help="Folder containing images to be parsed")
    args = parser.parse_args()
    dataset = CustomDataset(args.image_folder_name)
    dataloader = torch.utils.data.DataLoader(dataset)
    net = Inception3Softmax()
    softmax_scores = []
    iou_scores = []
    with torch.no_grad():
        for image, tgt, score in dataloader:
            softmaxes = net.forward(image)
            softmax_scores.append(softmaxes[tgt])
            iou_scores.append(score)
    plt.plot(iou_scores, softmax_scores)
    plt.title("Softmax probability of correct class vs. IoU score")
    plt.show()


if __name__ == "__main__":
    main()

