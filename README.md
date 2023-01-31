# Aesthetic Scorer extension for SD Automatic WebUI

Uses existing CLiP model with an additional small pretrained model to calculate perceived aesthetic score of an image  

Enable or disable via `Settings` -> `Aesthetic scorer`  

This is an *"invisible"* extension, it runs in the background before any image save and  
appends **`score`** as *PNG info section* and/or *EXIF comments* field

## Notes

- Configuration via **Settings** &rarr; **Aesthetic scorer**  
  ![screenshot](aesthetic-scorer.jpg)
- Extension obeys existing **Move VAE and CLiP to RAM** settings
- Models will be auto-downloaded upon first usage (small)
- Score values are `0..10`  
- Supports both `CLiP-ViT-L/14` and `CLiP-ViT-B/16`

This extension uses different method than [Aesthetic Image Scorer](https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer) extension which:
- Uses modified [SD Chad scorer](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/1831) implementation
- Windows-only!
- Executes as to replace `image.save` so limited compatibity with other *non-txt2img* use-cases

## CLI

As a utility for batch processing, this extension can be used from CLI as well  
Input param provided can be image, list of images, wildcards or folder  
Score is only output to console and does not modify original file  

> python aesthetic-scorer-cli.py ~/generative/Demo/*.jpg

    Loading CLiP model: ViT-L/14
    Loading Aesthetic Score model: sac_public_2022_06_29_vit_l_14_linear.pth
    Aesthetic score: 5.0 for image /home/vlado/generative/Demo/abby.jpg
    Aesthetic score: 4.18 for image /home/vlado/generative/Demo/ana.jpg
    Aesthetic score: 4.12 for image /home/vlado/generative/Demo/dreamkelly.jpg

## Credits

- Based on: [simulacra-aesthetic-models](https://github.com/crowsonkb/simulacra-aesthetic-models)  
- Training data set: [simulacra-aesthetic-captions](https://github.com/JD-P/simulacra-aesthetic-captions)  
