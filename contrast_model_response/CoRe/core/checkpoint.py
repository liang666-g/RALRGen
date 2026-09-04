# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:06:43 2020

@author: hasee
"""
import configuration
import os
from configuration import encoder_config, runfig
import torch


def load_checkpoint(checkpoint_path):
    # It's weird that if `map_location` is not given, it will be extremely slow.
    # try:
    #     torch._utils._rebuild_tensor_v2
    # except AttributeError:
    #     def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    #         tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
    #         tensor.requires_grad = requires_grad
    #         tensor._backward_hooks = backward_hooks
    #         return tensor
    #
    #     torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    return torch.load(checkpoint_path, map_location=lambda storage, loc: storage)


def save_checkpoint(experiment_name, encoder, decoder, encoder_optim, decoder_optim,
                    total_accuracy, total_loss, global_step):
    experiment_name = experiment_name.replace(':', '-').replace('/', '-').replace('\\', '-')

    checkpoint = {
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'encoder_optim_state_dict': encoder_optim.state_dict(),
        'decoder_optim_state_dict': decoder_optim.state_dict()
    }

    checkpoint_path = (f"./data/checkpoints/wc_{runfig.word_vec_size}_hs_{encoder_config.hidden_size}_ln_{encoder_config.num_layers}_dp_{encoder_config.dropout:.1f}"
                       f"/{experiment_name}_acc_{total_accuracy:.2f}_loss_{total_loss:.2f}_step_{global_step}.pt")

    directory, filename = os.path.split(os.path.abspath(checkpoint_path))

    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        if total_accuracy < 70:
            files = os.listdir(directory)
            for f in files:
                os.remove(os.path.join(directory, f))

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def get_checkpoint(checkpoint_name):
    out_dir = './data/checkpoints/'
    checkpoint_path = os.path.join(out_dir, checkpoint_name)
    print("Current checkpoint path is ", checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    print('=' * 100)
    return checkpoint
