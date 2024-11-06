# generate_captions.py
import os
import json
import torch
import clip
import skimage.io as io
import PIL.Image
from transformers import GPT2Tokenizer

import config
from inference import generate_beam, generate2
from model import ClipCaptionModel, ClipCaptionPrefix, MappingType
from tqdm import tqdm
import argparse

import warnings
warnings.filterwarnings("ignore", message=".*not compiled with flash attention.*")



# Helper function to parse model filename
def parse_model_filename(model_filename):
    # Extract dataset name (handle cases like 'sydney_cn')
    parts = model_filename.split('_')
    if parts[1] == 'cn':
        dataset_name = f"{parts[0]}_cn"
    else:
        dataset_name = parts[0]

    # Extract mapping type by checking substrings
    if 'mlp' in model_filename:
        mapping_type = MappingType.MLP
    elif 'transformer' in model_filename:
        mapping_type = MappingType.Transformer
    elif 'amht' in model_filename:
        mapping_type = MappingType.AdaptiveTransformer
    else:
        raise ValueError(f"Unknown mapping type in model filename: {model_filename}")

    # Return the extracted dataset name and mapping type
    return dataset_name, mapping_type




def generate_captions_for_sydney(annotation_path, image_dir, output_file, model_path, use_beam_search=False,
                                 mapping_type=MappingType.Transformer):
    device = torch.device('cuda:0')
    CPU = torch.device('cpu')
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = ClipCaptionModel(prefix_length=config.prefix_length,
                             clip_length=config.prefix_length,
                             prefix_size=512,
                             num_layers=config.num_trans_layers,
                             mapping_type=mapping_type)
    model.load_state_dict(torch.load(model_path, map_location=CPU))
    model = model.eval().to(device)

    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    results = {}
    for annotation in tqdm(annotations['images'], desc="Generating Captions"):
        if annotation['split'] != 'val':
            continue
        image_id = annotation['imgid']
        img_path = os.path.join(image_dir, annotation['filename'])
        image = io.imread(img_path)
        pil_image = PIL.Image.fromarray(image)
        image = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
            prefix_embed = model.clip_project(prefix).reshape(1, config.prefix_length, -1)

        if use_beam_search:
            caption = generate_beam(model, tokenizer, embed=prefix_embed)[0]
            print('use beam_search')
        else:
            # print('use top-p generate2')
            caption = generate2(model, tokenizer, embed=prefix_embed)

        results[image_id] = caption

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(results, f, indent=4)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)  # ensure_ascii=False 保证中文字符不转义

    print(f'Captions saved to {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate captions for datasets (Sydney, UCM, RSICD).')

    # Only one argument, model, which is the model filename
    parser.add_argument('--model', type=str, default="sydney_cn_mlp.pt",
                        help='Model weights filename (e.g., sydney_mlp-009.pt, sydney_mlp_gpt-009.pt).')

    args = parser.parse_args()

    # Parse model filename to extract dataset and mapping type
    dataset_name, mapping_type = parse_model_filename(args.model)

    # Based on the dataset, set paths for annotations and images
    if dataset_name == 'sydney':
        annotation_path = './datasets/Sydney_captions/dataset.json'
        image_dir = './datasets/Sydney_captions/imgs'
    elif dataset_name == 'ucm':
        annotation_path = './datasets/UCM_captions/dataset.json'
        image_dir = './datasets/UCM_captions/imgs'
    elif dataset_name == 'rsicd':
        annotation_path = './datasets/RSICD/annotations_rsicd/dataset_rsicd.json'
        image_dir = './datasets/RSICD/RSICD_images'
    elif dataset_name == 'sydney_cn':
        annotation_path = './datasets/Sydney_captions/dataset_sydney_cn.json'
        image_dir = './datasets/Sydney_captions/imgs'
    elif dataset_name == 'ucm_cn':
        annotation_path = '../../datasets/UCM_captions/dataset_ucm_cn.json'
        image_dir = './datasets/UCM_captions/imgs'
    elif dataset_name == 'rsicd_cn':
        annotation_path = '../../datasets/RSICD/annotations_rsicd/dataset_rsicd_updated.json'
        image_dir = './datasets/RSICD/RSICD_images'
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    # Generate output_file path by replacing '.pt' with '.json'
    output_file = f'./output/eval_files/{args.model.replace(".pt", ".json")}'

    # Model path
    model_path = f'./output/model_weights/{args.model}'

    # Generate captions
    generate_captions_for_sydney(annotation_path, image_dir, output_file, model_path, mapping_type=mapping_type)
