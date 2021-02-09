from json import encoder
import torch.optim as optim
from .MRIBoneNet import MRIBoneNet
from .MRIJointNet import MRIJointNet

model_list = ['UNET3D', 'DENSENET1', "UNET2D", 'DENSENET2', 'DENSENET3', 'HYPERDENSENET', "SKIPDENSENET3D",
              "DENSEVOXELNET", 'VNET', 'VNET2', "RESNET3DVAE", "RESNETMED3D", "COVIDNET1", "COVIDNET2", "CNN",
              "HIGHRESNET", "MRIBONENET", "UNET3D", "MRIJOINTNET"]


def create_model(args):
    model_name = args.model
    assert model_name in model_list
    optimizer_name = args.opt
    lr = args.lr
    in_channels = args.inChannels
    num_classes = args.classes
    if hasattr(args, "joints"):
        num_heatmaps = args.joints
    else:
        num_heatmaps = 0
    weight_decay = 3e-5
    print("Building Model . . . . . . . ." + model_name)

    if model_name == "MRIBONENET":
        model = MRIBoneNet(in_channels=in_channels, classes=num_classes, seg_only=args.segonly, seg_net=args.segnet,
                            center_idx=args.joint_center_idx, use_lbs=args.use_lbs, encoder_only=args.encoderonly)
    elif model_name == "MRIJOINTNET":
        model = MRIJointNet(in_channels=in_channels, n_heatmaps=num_heatmaps, method=args.model_method)
        weight_decay = args.weight_reg

    print(model_name, 'Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.99, nesterov=True)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, optimizer
