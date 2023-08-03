import torch
import pytorch_lightning as pl
from model_persp import Encoder, Decoder, TypeDecoder
from typing import Optional


class CombinedLayoutNetPersp(pl.LightningModule):

    def __init__(
            self,
            flip: bool = False
    ):
        super().__init__()

        # There is no ResNet in LayoutNetV1 as it was introduced in LayouetNetV2 to replace SegNet
        self.encoder = Encoder(3)
        self.edge_decoder = Decoder(skip_num=2, out_planes=3)
        self.corner_decoder = Decoder(skip_num=3, out_planes=8)
        self.type_decoder = TypeDecoder()

        self.example_input_array = torch.randn(1, 3, 512, 512)

    def forward(self, x):
        # TODO: if self.flip then concat with flipped version and post-process using mean, just as we do in Lua and viz version

        en_list = self.encoder(x)
        edg_de_list = self.edge_decoder(en_list[::-1])
        cor_de_list = self.corner_decoder(en_list[-1:] + edg_de_list[:-1])
        type_tensor = self.type_decoder(en_list)

        # return (edg_de_list, cor_de_list, type_tensor)
        edg_tensor = torch.sigmoid(edg_de_list[-1])
        cor_tensor = torch.sigmoid(cor_de_list[-1])
        type_tensor = type_tensor.softmax(1)

        return edg_tensor, cor_tensor, type_tensor

    def load_component_weights(
            self,
            encoder_checkpoint_path: Optional[str],
            edge_decoder_checkpoint_path: Optional[str],
            corner_decoder_checkpoint_path: Optional[str],
            type_decoder_checkpoint_path: Optional[str]
    ):
        if encoder_checkpoint_path is not None:
            print('Loading encoder weights')
            assert self.encoder.load_state_dict(
                torch.load('./ckpt/pre_encoder.pth'))

        if edge_decoder_checkpoint_path is not None:
            print('Loading edge decoder weights')
            assert self.edge_decoder.load_state_dict(
                torch.load('./ckpt/pre_edg_decoder.pth'))

        if corner_decoder_checkpoint_path is not None:
            print('Loading corner decoder weights')
            assert self.corner_decoder.load_state_dict(
                torch.load('./ckpt/pre_cor_decoder.pth'))

        if type_decoder_checkpoint_path is not None:
            print('Loading type decoder weights')
            assert self.type_decoder.load_state_dict(
                torch.load('./ckpt/pre_type_decoder.pth'))
