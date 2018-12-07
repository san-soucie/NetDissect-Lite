import torch
import torch.nn as nn
import torchvision
import os
import argparse
import sys
import numpy as np
from scipy.stats import linregress

from matplotlib import pyplot as plt
from process_synthesized_images import NetSoftmax, get_params, CustomDataset
sys.path.insert(0,"../nets/cornet/")
import cornet_r, cornet_s, cornet_z

torch.backends.cudnn.deterministic = True
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

no_normalize = torchvision.transforms.Lambda(
    lambda t: torchvision.transforms.Normalize(mean=(np.multiply(np.mean(t.numpy(), axis=(1,2)),
                                                                 np.array([0.229, 0.224, 0.225])) -
                                                     np.multiply(np.std(t.numpy(), axis=(1,2)),
                                                                 np.array([0.485, 0.456, 0.406]))),
                                               std=np.std(t.numpy(), axis=(1, 2)).tolist())(t))
multiply_by_sigma = torchvision.transforms.Lambda(lambda t: t * torch.Tensor([0.229, 0.224, 0.225]).reshape([3,1,1]))

trnsfrm_norm = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
trnsfrm_no_norm = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  no_normalize,
                                                  multiply_by_sigma])

cornets_dataset = CustomDataset("../outputs/synthesizing-06:53:19/CORnet-S/clean/",
                                transform=trnsfrm_norm,
                                stopwords={})
cornetz_dataset = CustomDataset("../outputs/synthesizing-06:53:19/CORnet-Z/clean/",
                                transform=trnsfrm_norm,
                                stopwords={})

cornets_dataset_no_norm = CustomDataset("../outputs/synthesizing-06:53:19/CORnet-S/clean/",
                                transform=trnsfrm_no_norm ,
                                stopwords={})
cornetz_dataset_no_norm = CustomDataset("../outputs/synthesizing-06:53:19/CORnet-Z/clean/",
                                transform=trnsfrm_no_norm ,
                                stopwords={})

cornets_dataloader = torch.utils.data.DataLoader(cornets_dataset)
cornetz_dataloader = torch.utils.data.DataLoader(cornetz_dataset)
cornets_dataloader_no_norm = torch.utils.data.DataLoader(cornets_dataset_no_norm)
cornetz_dataloader_no_norm = torch.utils.data.DataLoader(cornetz_dataset_no_norm)

net = NetSoftmax()
cornet_s = cornet_s.CORnet_S()
cornet_s.load_state_dict({k[7:]: v for k, v in torch.load("../zoo/cornet_s_epoch43.pth.tar")["state_dict"].items()})
cornet_z = cornet_z.CORnet_Z()
cornet_z.load_state_dict({k[7:]: v for k, v in torch.load("../zoo/cornet_z_epoch25.pth.tar")["state_dict"].items()})
net.eval()
cornet_s.eval()
cornet_z.eval()


softmax_scores_object = []
iou_scores_object = []
cornet_scores_object = []
cornet_max_index_object = []
softmax_scores_bad = []
iou_scores_bad = []
cornet_scores_bad = []
cornet_max_index_bad= []
softmax_scores_texture = []
iou_scores_texture = []
cornet_scores_texture= []
cornet_max_index_texture = []

with torch.no_grad():
    cornet_model = cornet_s.cuda()
    cornet_dataloader = cornets_dataloader
    cornet_dataloader_no_norm = cornets_dataloader_no_norm

    net = net.cuda()
    for no_norm, norm in zip(cornet_dataloader_no_norm, cornet_dataloader):
        image_no_norm, score, path = no_norm
        image_norm, _, _ = norm
        data_no_norm = image_no_norm.cuda()
        data_norm = image_norm.cuda()
        cornet_softmax = net.forward(data_no_norm).cpu()
        softmaxes = net.forward(data_norm).cpu()
        val, idx = cornet_softmax.squeeze().max(0)
        if '_bad.' in path[0]:
            cornet_scores_bad.append(float(val))
            iou_scores_bad.append(score[0].item())
            cornet_max_index_bad.append(int(idx))
            softmax_scores_bad.append(softmaxes.tolist()[0][idx])
        elif "_object_" in path[0]:
            cornet_scores_object.append(float(val))
            iou_scores_object.append(score[0].item())
            cornet_max_index_object.append(idx)
            softmax_scores_object.append(softmaxes.tolist()[0][idx])
        else:
            cornet_scores_texture.append(float(val))
            iou_scores_texture.append(score[0].item())
            cornet_max_index_texture.append(idx)
            softmax_scores_texture.append(softmaxes.tolist()[0][idx])

plt.plot(iou_scores_object, softmax_scores_object, 'bo', label="Top-5 Unit (Object)")
plt.plot(iou_scores_texture, softmax_scores_texture, 'go', label="Top-5 Unit (Texture)")
plt.plot(iou_scores_bad, softmax_scores_bad, 'ro', label="Bottom-5 Unit")

gradient, intercept, r_value, p_value, std_err = linregress(
    iou_scores_bad + iou_scores_object + iou_scores_texture,
    softmax_scores_bad + softmax_scores_object + softmax_scores_texture)

xs = np.linspace(0.01, 0.99, 98)
ys = gradient * xs + intercept
plt.plot(xs, ys, c='k')
plt.gca().annotate("r = %3.5f\np = %3.5f" % (r_value, p_value), (0.01, 0.9))

plt.title("AlexNet Softmax of CORnet-S Maximal Activators")

plt.xlim(left=0, right=0.5)
plt.ylim(bottom=-0.1, top=1)
plt.xlabel('CORnet-S Unit IoU Score')
plt.ylabel('AlexNet Softmax')
plt.legend()
img_filename = "CORnet-S_Alexnet_softmax_vs_iou.pdf"
plt.savefig(img_filename, bbox_inches='tight')


softmax_scores_object = []
iou_scores_object = []
cornet_scores_object = []
cornet_max_index_object = []
softmax_scores_bad = []
iou_scores_bad = []
cornet_scores_bad = []
cornet_max_index_bad= []
softmax_scores_texture = []
iou_scores_texture = []
cornet_scores_texture= []
cornet_max_index_texture = []

with torch.no_grad():
    cornet_model = cornet_z.cuda()
    cornet_dataloader = cornetz_dataloader
    cornet_dataloader_no_norm = cornetz_dataloader_no_norm

    net = net.cuda()
    for no_norm, norm in zip(cornet_dataloader_no_norm, cornet_dataloader):
        image_no_norm, score, path = no_norm
        image_norm, _, _ = norm
        data_no_norm = image_no_norm.cuda()
        data_norm = image_norm.cuda()
        cornet_softmax = net.forward(data_no_norm).cpu()
        softmaxes = net.forward(data_norm).cpu()
        val, idx = cornet_softmax.squeeze().max(0)

        if '_bad.' in path[0]:
            cornet_scores_bad.append(float(val))
            iou_scores_bad.append(score[0].item())
            cornet_max_index_bad.append(int(idx))
            softmax_scores_bad.append(softmaxes.tolist()[0][idx])
        elif "_object_" in path[0]:
            cornet_scores_object.append(float(val))
            iou_scores_object.append(score[0].item())
            cornet_max_index_object.append(idx)
            softmax_scores_object.append(softmaxes.tolist()[0][idx])
        else:
            cornet_scores_texture.append(float(val))
            iou_scores_texture.append(score[0].item())
            cornet_max_index_texture.append(idx)
            softmax_scores_texture.append(softmaxes.tolist()[0][idx])

plt.clf()
plt.plot(iou_scores_object, softmax_scores_object, 'bo', label="Top-5 Unit (Object)")
plt.plot(iou_scores_texture, softmax_scores_texture, 'go', label="Top-5 Unit (Texture)")
plt.plot(iou_scores_bad, softmax_scores_bad, 'ro', label="Bottom-5 Unit")

gradient, intercept, r_value, p_value, std_err = linregress(
    iou_scores_bad + iou_scores_object + iou_scores_texture,
    softmax_scores_bad + softmax_scores_object + softmax_scores_texture)

xs = np.linspace(0.01, 0.99, 98)
ys = gradient * xs + intercept
plt.plot(xs, ys, c='k')
plt.gca().annotate("r = %3.5f\np = %3.5f" % (r_value, p_value), (0.01, 0.9))

plt.title("AlexNet Softmax of CORnet-Z Maximal Activators")

plt.xlim(left=0, right=0.5)
plt.ylim(bottom=-0.1, top=1)
plt.xlabel('CORnet-Z Unit IoU Score')
plt.ylabel('AlexNet Softmax')
plt.legend()
img_filename = "CORnet-Z_Alexnet_softmax_vs_iou.pdf"
plt.savefig(img_filename, bbox_inches='tight')

