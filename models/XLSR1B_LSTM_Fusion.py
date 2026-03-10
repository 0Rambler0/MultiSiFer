import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from s3prl.nn import S3PRLUpstream, Featurizer
from peft import LoraConfig, get_peft_model
from .LCNN import BLSTMLayer

class Model(nn.Module):
    def __init__(self, args, sr_model, device):
        super().__init__()
        self.device = device
        
        #LSTM parameters 
        input_size = 256
        hidden_size = 402

        ####
        # create network wav2vec 2.0
        ####
        self.upstream_model = S3PRLUpstream(name=sr_model)
        if args["is_lora"] == "True":
            print("lora active")
            target_modules = []
            for i in range(args['lora_layers'][0], args['lora_layers'][1]):
                for modules in args['inject_modules']:
                    target_modules.append('layers.{}.self_attn.{}'.format(i, modules))

            config = LoraConfig(
            target_modules=target_modules,
            bias='none')
            self.upstream_model = get_peft_model(self.upstream_model, config)
        # for p in self.parameters():
        #     p.requires_grad = False
        # for p in self.upstream_model.upstream.model.feature_extractor.parameters():
        #     p.requires_grad = True
        # for i in range(0, 5):
        #     for p in self.upstream_model.upstream.model.encoder.layers[i].parameters():
        #         p.requires_grad = True
        if args["layers"][0] == -1:
            layers_id = [self.upstream_model.num_layers-1]
        elif args["layers"][1] == -1:
            layers_id = list(range(args["layers"][0],self.upstream_model.num_layers))
        else:
            layers_id = list(range(args["layers"][0],args["layers"][1]))
        print("layers id:{}".format(layers_id))
        self.no_featurizer = True if len(layers_id)==1 else False
        if self.no_featurizer:
            self.layer = layers_id[0]
        else:
            self.featurizer = Featurizer(self.upstream_model, layers_id)
        
        # self.LL = nn.Linear(1024, 256)
        # self.selu = nn.SELU(inplace=True)

        self.lstm = nn.Sequential(
                nn.Linear(1280, input_size),
                BLSTMLayer(input_size, hidden_size),
                BLSTMLayer(hidden_size, hidden_size)
        )

        self.out_layer = nn.Linear(hidden_size, 2)

    def forward(self, x, x_len):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        all_hs, all_hs_len = self.upstream_model(x, x_len)
        if self.no_featurizer:
            hs = all_hs[self.layer]
        else:
            hs, hs_len = self.featurizer(all_hs, all_hs_len) 

        # x = self.LL(hs)
        # x = self.selu(x)
        # print(x.shape)
        x = self.lstm(hs)
        x = x.mean(1)

        output = self.out_layer(x)
        
        return output
