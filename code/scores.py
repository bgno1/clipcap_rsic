import json
import os
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice
from transformers import GPT2Tokenizer
import jieba
from nltk.tokenize import word_tokenize
import argparse
from compute_cider import compute_cider

import nltk
import string

remove_punctuation = False

# Function to remove punctuation from a list of tokens
def remove_punctuation_from_tokens(tokens):
    return [token for token in tokens if token not in string.punctuation and token != '，']

# nltk.download('punkt')

def tokenize_ptb(text):
    return word_tokenize(text)


def generate_output_filename(eval_file):

    dataset_options = ['sydney_cn', 'sydney', 'ucm_cn', 'ucm',  'rsicd_cn', 'rsicd' ]
    mapping_networks = ['mlp', 'transformer', 'r50', 'r101', 'amht']

    # dataset type
    dataset = next((option for option in dataset_options if option in eval_file), None)
    if not dataset:
        raise ValueError("The evaluation file does not contain a recognized dataset name.")

    # mapping network type
    mapping_network = next((network for network in mapping_networks if network in eval_file), None)
    if not mapping_network:
        raise ValueError("The evaluation file does not contain a recognized mapping network type.")

    gpt_suffix = '_gpt' if 'gpt' in eval_file else ''
    output_file_name = f'{dataset}_{mapping_network}{gpt_suffix}.txt'
    return output_file_name


def main(eval_file):
    dataset_options = ['sydney_cn', 'sydney', 'ucm_cn', 'ucm',  'rsicd_cn', 'rsicd']
    dataset = next((option for option in dataset_options if option in eval_file), None)
    if not dataset:
        raise ValueError("The evaluation file does not match any supported dataset names.")

    generated_path = os.path.join('./output/eval_files', eval_file)

    output_dir = './output/scores'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, generate_output_filename(eval_file))

    if 'sydney' in dataset:
        reference_path = './datasets/Sydney_captions/dataset.json' if 'cn' not in dataset \
            else './datasets/Sydney_captions/dataset_sydney_cn.json'
    elif 'ucm' in dataset:
        reference_path = './datasets/UCM_captions/dataset.json' if 'cn' not in dataset \
            else './datasets/UCM_captions/dataset_ucm_cn.json'
    elif 'rsicd' in dataset:
        reference_path = './datasets/RSICD/annotations_rsicd/dataset_rsicd.json' if 'cn' not in dataset \
            else './datasets/RSICD/annotations_rsicd/dataset_rsicd_cn.json'
    print('dataset:',dataset,'\treference_path:',reference_path)

    tokenizer_choice = 'jieba'  # 'split', 'ptb', 'gpt2', 'jieba'
    use_smooth_function = True
    bleu_choice = 'corpus'  # 'sentence', 'corpus'

    # Load ground truth annotations
    with open(reference_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    # Load generated captions
    with open(generated_path, 'r', encoding='utf-8') as f:
        generated_captions = json.load(f)

    bleu_scores = {1: [], 2: [], 3: [], 4: []}
    references = []
    candidates = []
    gts = {}
    res = {}

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    for image_id, generated_caption in tqdm(generated_captions.items(), desc="Processing captions"):
        ground_truth_captions = [sentence['raw'] if 'cn' not in dataset else sentence['raw_chinese'] for annotation in
                                 annotations['images'] if int(annotation['imgid']) == int(image_id) for sentence in
                                 annotation['sentences']]

        if tokenizer_choice == 'split':
            reference_splits = [ref.split() for ref in ground_truth_captions]
            candidate_splits = generated_caption.split()
        elif tokenizer_choice == 'jieba':
            reference_splits = [list(jieba.cut(ref)) for ref in ground_truth_captions]
            candidate_splits = list(jieba.cut(generated_caption))
        elif tokenizer_choice == 'gpt2':
            reference_splits = [tokenizer.tokenize(ref) for ref in ground_truth_captions]
            candidate_splits = tokenizer.tokenize(generated_caption)
        elif tokenizer_choice == 'ptb':
            reference_splits = [tokenize_ptb(ref) for ref in ground_truth_captions]
            candidate_splits = tokenize_ptb(generated_caption)

        # If remove_punctuation is True, remove punctuation from splits
        if remove_punctuation:
            reference_splits = [remove_punctuation_from_tokens(ref) for ref in reference_splits]
            candidate_splits = remove_punctuation_from_tokens(candidate_splits)

        references.append(reference_splits)
        candidates.append(candidate_splits)
        print('refs: ', references)
        print('cands: ', candidates)

        gts[image_id] = [{'caption': ref} for ref in ground_truth_captions]
        res[image_id] = [{'caption': generated_caption}]

    smooth_fn = SmoothingFunction().method1 if use_smooth_function else None
    if bleu_choice == 'corpus':
        bleu_1_score = corpus_bleu(references, candidates, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
        bleu_2_score = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
        bleu_3_score = corpus_bleu(references, candidates, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn)
        bleu_4_score = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25),
                                   smoothing_function=smooth_fn)
    else:
        bleu_1_score = bleu_2_score = bleu_3_score = bleu_4_score = 0
        total = len(candidates)
        for ref, cand in zip(references, candidates):
            bleu_1_score += sentence_bleu(ref, cand, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
            bleu_2_score += sentence_bleu(ref, cand, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
            bleu_3_score += sentence_bleu(ref, cand, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn)
            bleu_4_score += sentence_bleu(ref, cand, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
        bleu_1_score /= total
        bleu_2_score /= total
        bleu_3_score /= total
        bleu_4_score /= total

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    if '_cn' in dataset:
        print('calculating CIDEr-cn score for Chinese sentences.')
        _, cider_score = compute_cider(gts, res)
    else:
        print('calculating CIDEr score for English sentences.')
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gts, res)

    meteor_scorer = Meteor()
    meteor_score, _ = meteor_scorer.compute_score(gts, res)

    spice_scorer = Spice()
    spice_score, _ = spice_scorer.compute_score(gts, res)

    scores = {
        'BLEU-1': bleu_1_score,
        'BLEU-2': bleu_2_score,
        'BLEU-3': bleu_3_score,
        'BLEU-4': bleu_4_score,
        'CIDEr': cider_score,
        'METEOR': meteor_score,
        'SPICE': spice_score
    }

    with open(output_file, 'w') as f:
        for metric, score in scores.items():
            print(f'{metric}: {score:.3f}')
            f.write(f'{metric}: {score:.3f}\n')
    print('scores saved to ', output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and evaluate captions for image datasets.')
    parser.add_argument('--eval_file', type=str, default='sydney_cn_amht.json',
                        help='The generated captions file to be evaluated.')
    args = parser.parse_args()
    main(args.eval_file)
