import os, sys, torch, torchvision, timeit
from torchvision import transforms
import matplotlib.pyplot as plt
sys.path.insert(0, "./processing/")
from process_synthesized_images import NetSoftmax
os.chdir("./dataset/tiny-imagenet-200/test/images")
no_softmax = False
net = NetSoftmax().cuda()
if no_softmax:
    net = torchvision.models.alexnet(pretrained=True).cuda()
resize = torchvision.transforms.Resize(224)
to_tensor = torchvision.transforms.ToTensor()
loader = torchvision.datasets.folder.pil_loader
num_files = len(os.listdir())
outputs = []
t = timeit.Timer()
start_time = t.timer()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
for i, file in enumerate(os.listdir()):
    with torch.no_grad():
        data = normalize(to_tensor(resize(loader(file)))).unsqueeze(0).cuda()
        if not no_softmax:
            outputs.append(float(net.forward(data).squeeze().cpu().max()))
        else:
            outputs.append(net.forward(data)
                              .squeeze()
                              .cpu()
                              .numpy())
        del data
        torch.cuda.empty_cache()
        if i % 100 == 0:
            curr_time = t.timer() - start_time
            print("Processed file %d/%d. Time elapsed: %3.5f. Estimated time remaining: %3.5f" %
                  (i+1, num_files, curr_time, (curr_time / (i+1)) * (10000 - (i+1)) ))
if not no_softmax:
    plt.hist(outputs)
    plt.show()