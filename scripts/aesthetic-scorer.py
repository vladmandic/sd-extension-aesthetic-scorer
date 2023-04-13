import os
from inspect import getsourcefile

import gradio as gr
import requests
import torch
import clip
from torch import nn
from torch.nn import functional as f
from torchvision import transforms
from torchvision.transforms import functional as tf

from modules import devices, script_callbacks, shared
from modules.script_callbacks import ImageSaveParams


# extension_path = 'extensions/sd-extension-aesthetic-scorer'
extension_path = os.path.join(os.path.dirname(getsourcefile(lambda:0)), '..')
git_home = 'https://github.com/vladmandic/sd-extensions/blob/main/extensions/aesthetic-scorer/models'
error = False
clip_model = None
aesthetic_model = None
normalize = transforms.Normalize(mean = [0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])


class AestheticMeanPredictionLinearModel(nn.Module):
    def __init__(self, feats_in):
        super().__init__()
        self.linear = nn.Linear(feats_in, 1)

    def forward(self, tensor):
        x = f.normalize(tensor, dim=-1) * tensor.shape[-1] ** 0.5
        return self.linear(x)


def find_model():
    global error # pylint: disable=global-statement
    if shared.opts.aesthetic_scorer_clip_model == 'ViT-L/14':
        model_name = 'sac_public_2022_06_29_vit_l_14_linear.pth'
    elif shared.opts.aesthetic_scorer_clip_model == 'ViT-B/16':
        model_name = 'sac_public_2022_06_29_vit_b_16_linear.pth'
    else:
        model_name = shared.opts.aesthetic_scorer_clip_model
        print(f'Aesthetic scorer: cannot match model for CLiP model {shared.opts.aesthetic_scorer_clip_model}')
        error = True
    model_path = os.path.join(extension_path, 'models', model_name)

    if not error and not os.path.exists(model_path):
        try:
            print(f'Aesthetic scorer downloading model: {model_name}')
            url = f"{git_home}/{model_name}?raw=true"
            r = requests.get(url, timeout=60)
            with open(model_path, "wb") as file:
                file.write(r.content)
        except Exception as e:
            print(f'Aesthetic scorer downloading model failed: {model_name}:', e)

    return model_path


def load_models():
    global clip_model # pylint: disable=global-statement
    global aesthetic_model # pylint: disable=global-statement
    if clip_model is None:
        try:
            model_root = shared.opts.clip_models_path
        except:
            model_root = shared.cmd_opts.clip_models_path
        print(f'Loading CLiP model {shared.opts.aesthetic_scorer_clip_model} ')
        clip_model, _clip_preprocess = clip.load(shared.opts.aesthetic_scorer_clip_model, jit = False, device = shared.device, download_root = model_root)
        clip_model.eval().requires_grad_(False)
        idx = torch.tensor(0).to(shared.device)
        first_embedding = clip_model.token_embedding(idx)
        expected_shape = first_embedding.shape[0]
    if aesthetic_model is None:
        aesthetic_model = AestheticMeanPredictionLinearModel(expected_shape)
        aesthetic_model.load_state_dict(torch.load(find_model()))
    # move to gpu
    clip_model = clip_model.to(shared.device)
    aesthetic_model = aesthetic_model.to(shared.device)
    return


def cleanup_models():
    global clip_model # pylint: disable=global-statement
    global aesthetic_model # pylint: disable=global-statement
    if not shared.opts.interrogate_keep_models_in_memory:
        clip_model = clip_model.to(devices.cpu)
        aesthetic_model = aesthetic_model.to(devices.cpu)
    devices.torch_gc()
    return


def on_before_image_saved(params: ImageSaveParams):
    global error # pylint: disable=global-statement
    if not shared.opts.aesthetic_scorer_enabled or error or params.image is None: # dont try again if previously errored out or no image
        return params
    try:
        load_models()
        img = params.image.convert('RGB')
        img = tf.resize(img, 224, transforms.InterpolationMode.LANCZOS) # resizes smaller edge
        img = tf.center_crop(img, (224,224)) # center crop non-squared images
        img = tf.to_tensor(img).to(shared.device)
        img = normalize(img)
        clip_image_embed = f.normalize(clip_model.encode_image(img[None, ...]).float(), dim = -1)
        score = aesthetic_model(clip_image_embed)
        score = round(score.item(), 2)
        # print('Aesthetic score:', score)
        if 'parameters' in params.pnginfo:
            params.pnginfo['parameters'] += f', Score: {score}'
        cleanup_models()
    except Exception as e:
        print('Aesthetic scorer error:', e)
        error = True
    return params


def on_ui_settings():
    section = ('aesthetic_scorer', "Aesthetic scorer")
    shared.opts.add_option("aesthetic_scorer_enabled", shared.OptionInfo(
        default = False,
        label = "Enabled",
        component = gr.Checkbox,
        component_args = { 'interactive': True },
        section = section
    ))
    shared.opts.add_option("aesthetic_scorer_clip_model", shared.OptionInfo(
        default = 'ViT-L/14',
        label = "CLiP model",
        component = gr.Radio,
        component_args = { 'choices': ['ViT-L/14', 'ViT-B/16'] },
        section = section
    ))


script_callbacks.on_before_image_saved(on_before_image_saved)
script_callbacks.on_ui_settings(on_ui_settings)
