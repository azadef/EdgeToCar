import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torchvision import transforms

import utils
from transformer_net import TransformerNet


def stylize(input_image, style, cuda):
    device = torch.device("cuda" if cuda else "cpu")

    content_image = utils.load_image(input_image, scale=None)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load("saved_models/" + style + ".pth")
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image).cpu()
            
    utils.save_image("results/" + style + '_out.jpg', output[0])
    
def main():
    arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")

    arg_parser.add_argument("--input_image", type=str, required=True,
                                 help="path to content image you want to stylize")
    arg_parser.add_argument("--output_image", type=str, required=False,
                                 help="path for saving the output image")
    arg_parser.add_argument("--style", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    arg_parser.add_argument("--cuda", type=int, required=False,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    
    
    args = arg_parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    stylize(args.input_image, args.style, True)
    
    
if __name__ == "__main__":
    main()