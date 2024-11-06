import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse

import warnings
warnings.filterwarnings("ignore", message=".*not compiled with flash attention.*")


# Dictionary of dataset annotation paths
dataset_paths = {
    'sydney': './datasets/Sydney_captions/dataset.json',
    'ucm': './datasets/UCM_captions/dataset.json',
    'rsicd': './datasets/RSICD/annotations_rsicd/dataset_rsicd.json',
    'sydney_cn': './datasets/Sydney_captions/dataset_sydney_cn.json',
    'ucm_cn': './datasets/UCM_captions/dataset_ucm_cn.json',
    'rsicd_cn': './datasets/RSICD/annotations_rsicd/dataset_rsicd_cn.json'
}

# Dictionary of image directories for each dataset
image_dirs = {
    'sydney': './datasets/Sydney_captions/imgs',
    'ucm': './datasets/UCM_captions/imgs',
    'rsicd': './datasets/RSICD/RSICD_images',
    'sydney_cn': './datasets/Sydney_captions/imgs',
    'ucm_cn': './datasets/UCM_captions/imgs',
    'rsicd_cn': './datasets/RSICD/RSICD_images'
}


def main(clip_model_type, dataset_type):

    # Load device
    device = torch.device('cuda:0')
    if device.type == 'cuda':
        print(f"Using device: {device}, {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using device: {device}")

    # Set output directory
    clip_model_name = clip_model_type.replace('/', '_')  # "ViT-B/32" -> "ViT-B_32"
    data_dir = './data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    out_path = os.path.join(f'./data/clip_emb_{clip_model_name}_{dataset_type}.pkl')  # Output file path

    # Load CLIP model and preprocess function
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    # Load annotation file
    if dataset_type in dataset_paths:
        # with open(dataset_paths[dataset_type], 'r') as f:
        with open(dataset_paths[dataset_type], 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        # Construct training data
        data = []
        for img in raw_data['images']:
            if img['split'] == 'train':
                for sentence in img['sentences']:
                    data.append({
                        'image_id': img['imgid'],
                        'id': sentence['sentid'],
                        # 'caption': sentence['raw'],
                        'caption': sentence.get('raw_chinese', sentence['raw']),
                        'filename': img['filename']
                    })
        print(f"{len(data)} captions loaded from {dataset_type}-caption json (training set) ")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    all_embeddings = []
    all_captions = []

    # Iterate over each annotation
    for i in tqdm(range(len(data))):
        # dict of 'imgid', 'sentid',
        d = data[i]

        # Get image path
        if dataset_type in image_dirs:
            filename = os.path.join(image_dirs[dataset_type], d['filename'])
        else:
            raise ValueError(f"Unsupported dataset type for image loading: {dataset_type}")

        # Read image
        image = io.imread(filename)

        # Encode image using the CLIP model
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()   # Move to CPU for saving
        d["clip_embedding"] = i  # Index of the embedding
        all_embeddings.append(prefix)
        all_captions.append(d)

        # Save every 10,000 embeddings
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    # Save the final result
    with open(out_path, 'wb') as f:
        pickle.dump(
            {
                "clip_embedding": torch.cat(all_embeddings, dim=0),     # CLIP embedded prefixes
                "captions": all_captions}, f                            # captions
        )

    print('Done')
    print(f"{len(all_embeddings)} embeddings saved ")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--dataset_type', default="sydney", choices=('sydney', 'ucm', 'rsicd', 'sydney_cn', 'ucm_cn', 'rsicd_cn'))
    args = parser.parse_args()
    exit(main(args.clip_model_type, args.dataset_type))
