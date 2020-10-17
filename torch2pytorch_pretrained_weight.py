import argparse
import torchfile
import numpy as np

import torch

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--torch_pretrained', default='ckpt/panofull_lay_pretrained.t7',
                    help='path to load pretrained .t7 file')
parser.add_argument('--encoder', default='ckpt/pre_encoder.pth',
                    help='dump path. skip if not given')
parser.add_argument('--edg_decoder', default='ckpt/pre_edg_decoder.pth',
                    help='dump path. skip if not given')
parser.add_argument('--cor_decoder', default='ckpt/pre_cor_decoder.pth',
                    help='dump path. skip if not given')
parser.add_argument('--type_decoder', default='ckpt/pre_type_decoder.pth',
                    help='dump path. skip if not given')
parser.add_argument('--perspective', action='store_true')

args = parser.parse_args()
if args.perspective:
    from model_persp import Encoder, Decoder, TypeDecoder
else:
    from model import Encoder, Decoder

torch_pretrained = torchfile.load(args.torch_pretrained)
if args.encoder:
    in_dim = 3 if args.perspective else 6
    encoder = Encoder(in_dim)
if args.edg_decoder:
    edg_decoder = Decoder(skip_num=2, out_planes=3)
if args.cor_decoder:
    out_dim = 8 if args.perspective else 1
    cor_decoder = Decoder(skip_num=3, out_planes=8)
if args.type_decoder:
    assert(args.perspective)
    type_decoder = TypeDecoder()

# Check number of parameters
print('torch parameters num:', torch_pretrained.shape[0])
total_parameter = 0
if args.encoder:
    for p in encoder.parameters():
        total_parameter += np.prod(p.size())
if args.edg_decoder:
    for p in edg_decoder.parameters():
        total_parameter += np.prod(p.size())
if args.cor_decoder:
    for p in cor_decoder.parameters():
        total_parameter += np.prod(p.size())
if args.type_decoder:
    for p in type_decoder.parameters():
        total_parameter += np.prod(p.size())
print('pytorch model parameters num:', total_parameter)

assert torch_pretrained.shape[0] >= total_parameter, 'not enough weight to load'
if torch_pretrained.shape[0] > total_parameter:
    print('Note: fewer parameters then pretrained weights !!!')
    import ipdb as pdb; pdb.set_trace()
else:
    print('Number of parameters match!')

# Coping parameters
def copy_params(idx, parameters):
    for p in parameters:
        layer_p_num = np.prod(p.size())
        p.view(-1).copy_(torch.FloatTensor(
            torch_pretrained[idx:idx+layer_p_num]))
        idx += layer_p_num
        print('copy pointer current position: %d' % idx, end='\r', flush=True)
    return idx


print('# of parameters matched, start coping')
idx = 0
if args.encoder:
    idx = copy_params(idx, encoder.parameters())
    torch.save(encoder.state_dict(), args.encoder)
if args.edg_decoder:
    idx = copy_params(idx, edg_decoder.parameters())
    torch.save(edg_decoder.state_dict(), args.edg_decoder)
if args.cor_decoder:
    idx = copy_params(idx, cor_decoder.parameters())
    torch.save(cor_decoder.state_dict(), args.cor_decoder)
if args.type_decoder:
    idx = copy_params(idx, type_decoder.parameters())
    torch.save(type_decoder.state_dict(), args.type_decoder)


print('\nAll thing well done')
