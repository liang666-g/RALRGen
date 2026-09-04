import os
import torch
import torch._utils


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


def save_checkpoint(opts, experiment_name, encoder, decoder, encoder_optim, decoder_optim,
                    score, total_loss, global_step):
    checkpoint = {
        'opts': opts,
        'global_step': global_step,
        'score': score,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'encoder_optim_state_dict': encoder_optim.state_dict(),
        'decoder_optim_state_dict': decoder_optim.state_dict()
    }

    ext_opts = [
        int(opts.use_sent_rate),
        int(opts.use_sent_senti),
        int(opts.use_sent_len),
        int(opts.use_app_cate),
        int(opts.use_keyword),
        int(opts.tie_ext_feature)
    ]
    ext_opts = ''.join(map(str, ext_opts))

    checkpoint_root = getattr(opts, 'checkpoint_dir', './checkpoints')

    checkpoint_dir = os.path.join(
        checkpoint_root,
        'single_wc%s_hs%s_ln%s_dp%.1f_rslckt%s' % (
            str(opts.word_vec_size),
            str(opts.hidden_size),
            str(opts.num_layers),
            opts.dropout,
            ext_opts
        )
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        '%s_bleu_%.6f_loss_%.4f_step_%d.pt' % (
            experiment_name,
            score,
            total_loss,
            global_step
        )
    )

    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path