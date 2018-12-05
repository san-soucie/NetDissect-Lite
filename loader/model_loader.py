import settings
import torch
import torchvision
from nets.cornet.cornet_z import CORnet_Z
from nets.cornet.cornet_r import CORnet_R
from nets.cornet.cornet_s import CORnet_S
import pretrainedmodels # pip install pretrainedmodels

local_model_defs = { 'cornetz': CORnet_Z, 'cornetr': CORnet_R, 'cornets': CORnet_S }

def getmodel(name, pretrained=True, dataset='imagenet', num_classes=1000):
    if name in local_model_defs.keys():
        return local_model_defs[name]()
    elif pretrained:
        if name in pretrainedmodels.models.__dict__:
            return pretrainedmodels.__dict__[name](pretrained=dataset, num_classes=num_classes)
        elif name in torchvision.models.__dict__ and dataset == 'imagenet':
            return torchvision.models.__dict__[name](pretrained=True, num_classes=num_classes)
        else:
            raise FileNotFoundError('Failed to find pretrained model \'' + name + "'")
    else:
        if name in pretrainedmodels.__dict__:
            return pretrainedmodels.__dict__[name](pretrained=None, num_classes=num_classes)
        elif name in torchvision.models.__dict__:
            return torchvision.models.__dict__[name](pretrained=False, num_classes=num_classes)
        else:
            raise FileNotFoundError('Failed to find model \'' + name + "'")

def loadmodel(hook_fn):
    if settings.MODEL_FILE is None:
        model = getmodel(settings.MODEL, pretrained=True, dataset=settings.DATASET, num_classes=settings.NUM_CLASSES)
    else:
        checkpoint = torch.load(settings.MODEL_FILE)
        if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
            model = getmodel(settings.MODEL, pretrained=False, num_classes=settings.NUM_CLASSES)
            if settings.MODEL_PARALLEL:
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                    'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        else:
            model = checkpoint
    try:
        for name in settings.FEATURE_NAMES:
            model._modules.get(name).register_forward_hook(hook_fn)
    except AttributeError as ae:
        print('Failed to find FEATURE_NAMES for this model!')
        print('Perhaps you need to add an entry to settings.py for this model? Here are the modules:')
        print(model._modules)
        raise ae
    if settings.GPU:
        if not torch.cuda.is_available():
            print("CUDA is not availible. Running on CPU...")
        else:
            model.cuda()
    model.eval()
    return model
