#!/bin/env python

import os
import re
import pathlib
import argparse
from inspect import getsourcefile

import piexif
import requests
import torch
import clip
from torch import nn
from torch.nn import functional
from torchvision import transforms
from torchvision.transforms import functional as tf
from PIL import Image, JpegImagePlugin, PngImagePlugin
from exif import Exif

git_home = 'https://github.com/vladmandic/sd-extensions/blob/main/extensions/aesthetic-scorer/models'
clip_model = None
aesthetic_model = None
normalize = transforms.Normalize(mean = [0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AestheticMeanPredictionLinearModel(nn.Module):
    def __init__(self, feats_in):
        super().__init__()
        self.linear = nn.Linear(feats_in, 1)

    def forward(self, tensor):
        x = functional.normalize(tensor, dim=-1) * tensor.shape[-1] ** 0.5
        return self.linear(x)


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def find_model(params):
    model_path = os.path.join(os.path.dirname(getsourcefile(lambda:0)), 'models', params.model)
    if not os.path.exists(model_path):
        try:
            print(f'Aesthetic scorer downloading model: {model_path}')
            url = f"{git_home}/{params.model}?raw=true"
            r = requests.get(url, timeout=60)
            with open(model_path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f'Aesthetic scorer downloading model failed: {model_path}:', e)
    return model_path


def load_models(params):
    global clip_model
    global aesthetic_model
    if clip_model is None:
        print(f'Loading CLiP model: {params.clip} ')
        clip_model, _clip_preprocess = clip.load(params.clip, jit = False, device = device)
        clip_model.eval().requires_grad_(False)
        idx = torch.tensor(0).to(device)
        first_embedding = clip_model.token_embedding(idx)
        expected_shape = first_embedding.shape[0]
        aesthetic_model = AestheticMeanPredictionLinearModel(expected_shape)
        print(f'Loading Aesthetic Score model: {params.model} ')
        model_path = find_model(params)
        aesthetic_model.load_state_dict(torch.load(model_path))
        clip_model = clip_model.to(device)
        aesthetic_model = aesthetic_model.to(device)
    return


def aesthetic_score(fn, params):
    global clip_model
    global aesthetic_model
    try:
        img = Image.open(fn)
    except Exception as e:
        print('Aesthetic scorer failed to open image:', e)
        return 0
    load_models(params)
    exif = Exif(img)
    thumb = img.convert('RGB')
    thumb = tf.resize(thumb, 224, transforms.InterpolationMode.LANCZOS) # resizes smaller edge
    thumb = tf.center_crop(thumb, (224,224)) # center crop non-squared images
    thumb = tf.to_tensor(thumb).to(device)
    thumb = normalize(thumb)
    encoded = clip_model.encode_image(thumb[None, ...]).float()
    clip_image_embed = functional.normalize(encoded, dim = -1)
    score = aesthetic_model(clip_image_embed)
    score = round(score.item(), 2)
    if params.save is not None:
        # prepare metadata
        if exif.UserComment is None and exif.parameters is not None:
            exif.exif['UserComment'] = exif.parameters
            del exif.exif['parameters']
        if exif.UserComment is None:
            exif.exif['UserComment'] = f'Score: {score}'
        elif 'Score:' in exif.UserComment:
            exif.exif['UserComment'] = re.sub(r'Score: \d+.\d+', f'Score: {score}', exif.UserComment)
        else:
            exif.exif['UserComment'] += f', Score: {score}'

        if params.save != '#': # save to specified file
            if os.path.isdir(params.save):
                fn = os.path.join(params.save, os.path.basename(fn))
            else:
                fn = params.save
    
        ext = pathlib.Path(fn).suffix.lower()
        if ext == '.jpg' or ext == '.jpeg' or ext == '.webp':
            print('Saving image:', fn)
            img.save(fn, exif=exif.bytes(), quality=params.quality)
        elif ext == '.png':
            print('Saving image:', fn)
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text('parameters', exif.UserComment, zip=False)
            img.save(fn, pnginfo=pnginfo, quality=params.quality)
        else:
            print('Save image unknown format', fn)
    if params.exif:
        print('Metadata:', exif.exif)
    print(f'Aesthetic score: {score} for image {fn}')
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'generate model previews')
    parser.add_argument('--model', type = str, default = 'sac_public_2022_06_29_vit_l_14_linear.pth', required = False, help = 'Aesthetic score model')
    parser.add_argument('--clip', type = str, default = 'ViT-L/14', required = False, help = 'CLiP model')
    parser.add_argument('--exif', default = False, action='store_true', help = 'Show full image exif information')
    parser.add_argument('--save', nargs='?', default=None, const='#', help='Save image with score into original image or to specified file or folder')
    parser.add_argument('--quality', type = int, default = 80, required = False, help = 'Image quality when saving images')
    parser.add_argument('input', type = str, nargs = '*', help = 'Input image(s) or folder(s)')
    params = parser.parse_args()
    for fn in params.input:
        if os.path.isfile(fn):
            aesthetic_score(fn, params)
        elif os.path.isdir(fn):
            for root, dirs, files in os.walk(fn):
                for file in files:
                    aesthetic_score(os.path.join(root, file), params)
    torch_gc()
