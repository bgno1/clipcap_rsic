import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import torch.optim as optim
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union

import config
from model import ClipCaptionModel, ClipCaptionPrefix, MappingType
from data import ClipCocoDataset


def train(dataset: ClipCocoDataset, model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):
    # device
    device = torch.device('cuda:0')
    if device.type == 'cuda':
        print(f"Using device: {device}, {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using device: {device}")

    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )

    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tokens.flatten(),
                ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()

    # save trained model
    save_path = os.path.join(output_dir, f"{output_prefix}.pt")
    torch.save(
        model.state_dict(),
        save_path
    )
    print(f"Model weights saved to {save_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sydney', help='Name of the dataset (e.g., sydney, ucm, rsicd_cn, etc.)')
    parser.add_argument('--clip_model_type', default='ViT-B/32', help='Model type (default: ViT-B/32)')
    parser.add_argument('--out_dir', default='./output/model_weights')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--prefix_length', type=int, default=20)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=12)
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--train_gpt', type=lambda x: bool(str(x).lower() in ['true', '1']), nargs='?', const=True,
                        default=False, help='Set to True to train both prefix and GPT, False to train only prefix.')
    parser.add_argument('--mapping_type', type=str, default='amht', help='mlp/transformer/amht')
    parser.add_argument('--num_layers', type=int, default=16)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--num_heads', type=int, default=16)
    args = parser.parse_args()

    pkl_path = f'./data/clip_emb_{args.clip_model_type.replace("/", "_")}_{args.dataset}.pkl'
    # ClipCocoDataset
    dataset = ClipCocoDataset(pkl_path,
                              config.prefix_length,
                              normalize_prefix=config.normalize_prefix)
    prefix_dim = 640 if args.is_rn else 512

    # mapping type
    args.mapping_type = {'mlp': MappingType.MLP,
                         'transformer': MappingType.Transformer,
                         'amht': MappingType.AdaptiveTransformer}[args.mapping_type]
    print(f'Mapping type: {args.mapping_type}')

    prefix = f"{args.dataset}_{args.mapping_type.value}"

    if args.train_gpt:
        prefix += "_gpt"
    print(f"Generated prefix: {prefix}")


    model = ClipCaptionPrefix(prefix_length=config.prefix_length,
                              clip_length=config.prefix_length,
                              prefix_size=prefix_dim,
                              num_layers=args.num_layers,
                              mapping_type=args.mapping_type)
    sys.stdout.flush()

    train(dataset,
          model,
          args,
          output_dir=args.out_dir,
          output_prefix=prefix)


if __name__ == '__main__':
    main()

