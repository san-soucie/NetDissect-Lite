import torch
import torch.nn as nn
import torchvision
import os
import argparse
import sys

from matplotlib import pyplot as plt
from process_synthesized_images import NetSoftmax, get_params, CustomDataset
sys.path.insert(0,"../nets/cornet/")
import cornet_r, cornet_s, cornet_z
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trnsfrm = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])


cornets_dataset = CustomDataset("../outputs/synthesizing-06:53:19/CORnet-S/clean/",
                                transform=trnsfrm,
                                stopwords={})
cornetz_dataset = CustomDataset("../outputs/synthesizing-06:53:19/CORnet-Z/clean/",
                                transform=trnsfrm,
                                stopwords={})

cornets_dataset_no_norm = CustomDataset("../outputs/synthesizing-06:53:19/CORnet-S/clean/",
                                transform=torchvision.transforms.ToTensor(),
                                stopwords={})
cornetz_dataset_no_norm = CustomDataset("../outputs/synthesizing-06:53:19/CORnet-Z/clean/",
                                transform=torchvision.transforms.ToTensor(),
                                stopwords={})

cornets_dataloader = torch.utils.data.DataLoader(cornets_dataset)
cornetz_dataloader = torch.utils.data.DataLoader(cornetz_dataset)
cornets_dataloader_no_norm = torch.utils.data.DataLoader(cornets_dataset)
cornetz_dataloader_no_norm = torch.utils.data.DataLoader(cornetz_dataset)

net = NetSoftmax()
cornet_s = cornet_s.CORnet_S()
cornet_s.load_state_dict({k[7:]: v for k, v in torch.load("../zoo/cornet_s_epoch43.pth.tar")["state_dict"].items()})
cornet_z = cornet_z.CORnet_Z()
cornet_z.load_state_dict({k[7:]: v for k, v in torch.load("../zoo/cornet_z_epoch25.pth.tar")["state_dict"].items()})



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
    for image, score, path in cornet_dataloader_no_norm:
        data = image.cuda()
        cornet_softmax = net.forward(data).cpu()
        val, idx = cornet_softmax.squeeze().max(0)
        if '_bad.' in path[0]:
            cornet_scores_bad.append(float(val))
            iou_scores_bad.append(score[0].item())
            cornet_max_index_bad.append(int(idx))
        elif "_object_" in path[0]:
            cornet_scores_object.append(float(val))
            iou_scores_object.append(score[0].item())
            cornet_max_index_object.append(idx)
        else:
            cornet_scores_texture.append(float(val))
            iou_scores_texture.append(score[0].item())
            cornet_max_index_texture.append(idx)
    k = 0
    for i, blob in enumerate(cornet_dataloader):
        image, score, path = blob
        data = image.cuda()
        softmaxes = net.forward(data).cpu()
        if '_bad.' in path[0]:
            softmax_scores_bad.append(softmaxes.tolist()[0][cornet_max_index_bad[i]])
        elif "_object_" in path[0]:
            softmax_scores_object.append(softmaxes.tolist()[0][cornet_max_index_object[i]])
        else:
            softmax_scores_texture.append(softmaxes.tolist()[0][cornet_max_index_object[i]])

plt.plot(iou_scores_object, softmax_scores_object, 'bo', label="Top-5 Unit (Object)")
plt.plot(iou_scores_texture, softmax_scores_texture, 'go', label="Top-5 Unit (Texture)")
plt.plot(iou_scores_bad, softmax_scores_bad, 'ro', label="Bottom-5 Unit")

plt.title("Softmax probability of CORnet-S Class Identification vs. IoU score)")

plt.xlim(left=0, right=1)
plt.ylim(bottom=0, top=1)
plt.xlabel('Unit IoU Score')
plt.ylabel('Softmax Score')
plt.legend()
img_filename = ("CORnet-S_Alexnet_softmax_vs_iou.pdf")
plt.savefig(img_filename, bbox_inches='tight')

