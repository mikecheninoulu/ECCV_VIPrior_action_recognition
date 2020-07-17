from __future__ import division
import torch
from torch import nn
from models import buildmodel
import pdb

def generate_model( opt):
    assert opt.model in ['resnext', 'tc3d']
    assert opt.model_depth in [50,101]

    from models.buildmodel import get_fine_tuning_parameters


    if opt.model == 'tc3d':
        if opt.model_depth == 50:
            model = buildmodel.C3D_CDC50(
                    num_classes=opt.n_classes,
                    input_channels=opt.input_channels,
                    theta=opt.cdc_theta,
                    output_layers=opt.output_layers)
        else:
            model = buildmodel.C3D_CDC(
                    num_classes=opt.n_classes,
                    input_channels=opt.input_channels,
                    theta=opt.cdc_theta,
                    output_layers=opt.output_layers)

    if opt.model == 'resnext':
        if opt.model_depth == 50:
            model = buildmodel.resnet50(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.resnet_shortcut,
                    cardinality=opt.resnext_cardinality,
                    sample_size=opt.sample_size,
                    sample_duration=opt.sample_duration,
                    input_channels=opt.input_channels,
                    output_layers=opt.output_layers)
        else:
            model = buildmodel.resnet101(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.resnet_shortcut,
                    cardinality=opt.resnext_cardinality,
                    sample_size=opt.sample_size,
                    sample_duration=opt.sample_duration,
                    input_channels=opt.input_channels,
                    output_layers=opt.output_layers)

    model = model.cuda()
    model = nn.DataParallel(model)

    if opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)

        assert opt.arch == pretrain['arch']
        model.load_state_dict(pretrain['state_dict'])
        model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
        model.module.fc = model.module.fc.cuda()

        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
        return model, parameters

    return model, model.parameters()
