import os
import torch
import argparse
from IPython import embed

def surgery_loop(args, surgery):
    save_name = args.tar_name + '_' + ('remove' if args.method == 'remove' else 'surgery') + '.pth'
    save_path = os.path.join(args.save_dir, save_name)
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = torch.load(args.src_path)
    if 'scheduler' in ckpt:
        del ckpt['scheduler']
    if 'optimizer' in ckpt:
        del ckpt['optimizer']
    if 'iteration' in ckpt:
        ckpt['iteration'] = 0
    if args.method == 'remove':
        for param_name in args.param_name:
            del ckpt['model'][param_name + '.weight']
            if param_name + '.bias' in ckpt['model']:
                del ckpt['model'][param_name + '.bias']
    elif args.method == 'randinit':
        tar_sizes = [TAR_SIZE * 4, TAR_SIZE]
        for idx, (param_name, tar_size) in enumerate(zip(args.param_name, tar_sizes)):
            if 'bbox_pred' in param_name:
                surgery(param_name, True, tar_size, ckpt)
                surgery(param_name, False, tar_size, ckpt)
            else:
                pretrained_weight = ckpt['model'][param_name]
                prev_cls = pretrained_weight.size(0)
                feat_size = pretrained_weight.size(1)
                new_weight = torch.rand((tar_size, feat_size))
                torch.nn.init.normal_(new_weight)
                new_weight[:prev_cls] = pretrained_weight[:prev_cls]
                ckpt['model'][param_name] = new_weight
    else:
        raise NotImplementedError

    torch.save(ckpt, save_path)
    print('save changed ckpt to {}'.format(save_path))

def main(args):
    def surgery(param_name, is_weight, tar_size, ckpt):
        weight_name = param_name + ('.weight' if is_weight else '.bias')
        pretrained_weight = ckpt['model'][weight_name]
        prev_cls = pretrained_weight.size(0)
        if is_weight:
            feat_size = pretrained_weight.size(1)
            new_weight = torch.rand((tar_size, feat_size))
            torch.nn.init.normal_(new_weight, 0, 0.01)
        else:
            new_weight = torch.zeros(tar_size)
        new_weight[:prev_cls] = pretrained_weight[:prev_cls]
        ckpt['model'][weight_name] = new_weight

    surgery_loop(args, surgery)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dior', choices=['dior','nwpu'])
    parser.add_argument('--src-path', type=str, default='', help='Path to the main checkpoint')
    parser.add_argument('--save-dir', type=str, default='', required=True, help='Save directory')
    parser.add_argument('--method', choices=['remove', 'randinit'], required=True,
                        help = 'remove = remove the final layer of the base detector. '
                               'randinit = randomly initialize novel weights.')
    parser.add_argument('--param-name', type=str, nargs='+', help='Target parameter names',
                        default=['roi_heads.box_predictor.bbox_pred','roi_heads.cpln.representatives'])
    parser.add_argument('--tar-name', type=str, default='model_reset', help='Name of the new ckpt')
    args = parser.parse_args()

    if args.dataset == 'nwpu':
        TAR_SIZE = 10
    elif args.dataset == 'dior':
        TAR_SIZE = 20
    else:
        raise NotImplementedError
    main(args)