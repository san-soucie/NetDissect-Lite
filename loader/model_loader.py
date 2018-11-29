import settings
import torch
import torchvision
from nets.cornet.cornet_z import CORnet_Z
from nets.cornet.cornet_r import CORnet_R
from nets.cornet.cornet_s import CORnet_S

local_model_defs = { 'cornetz': CORnet_Z, 'cornetr': CORnet_R, 'cornets': CORnet_S }

def loadmodel(hook_fn):
    if settings.MODEL_FILE is None:
        model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
    else:
        checkpoint = torch.load(settings.MODEL_FILE)
        if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
            if settings.MODEL in local_model_defs.keys():
                model = local_model_defs[settings.MODEL]()
            else:
                model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
            if settings.MODEL_PARALLEL:
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                    'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        else:
            model = checkpoint
    for name in settings.FEATURE_NAMES:
        model._modules.get(name).register_forward_hook(hook_fn)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model
