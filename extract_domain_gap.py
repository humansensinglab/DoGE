import argparse
import os
from typing import *

import clip
import numpy as np
import torch
import torchvision
import nltk
from nltk.corpus import brown
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import (
    CLIPProcessor,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

clip_model = None


def is_folder_of_folders(dir: str) -> bool:
    return any(map(lambda x: os.path.isdir(os.path.join(dir, x)), os.listdir(dir)))


def data_loader_torchProc(
    dir: str, preprocess: torchvision.transforms.Compose, limit: int = 128
) -> torch.Tensor:
    # if dir is a folder of folders, use ImageFolder to load the images
    if is_folder_of_folders(dir):
        dataset = torchvision.datasets.ImageFolder(dir, transform=preprocess)
        return torch.stack(list(map(lambda x: x[0], dataset))[:limit])
    # if dir is a folder of images, load images directly
    else:
        return torch.stack(
            list(
                map(
                    lambda x: preprocess(Image.open(f"{dir}/{x}").convert("RGB")),
                    os.listdir(dir),
                )
            )[:limit]
        )


def data_loader_clipProc(
    dir: str, preprocess: CLIPProcessor, limit: int = 128
) -> torch.Tensor:
    src_set = []
    print(f"Loading images from {dir}... ", end="")
    # if dir is a folder of folders, draw images from each folder
    if is_folder_of_folders(dir):
        classes = list(
            filter(lambda x: os.path.isdir(os.path.join(dir, x)), os.listdir(dir))
        )
        np.random.shuffle(classes)
        alloc = ((limit + len(classes) - 1) // len(classes)) if limit > 0 else 65536
        #print(f"Loading {alloc} images per class for {len(classes)} classes")
        for class_name in classes:
            files = os.listdir(os.path.join(dir, class_name))
            np.random.shuffle(files)
            for f in files[:alloc]:
                with Image.open(os.path.join(dir, class_name, f)) as img:
                    src_set.append(img.copy())
        assert len(src_set) > 0, f"\nSource directory {dir} is empty"
        if len(src_set) < limit:
            print(f"\nWarning: Source directory {dir} has less than {limit} images")
        np.random.shuffle(src_set)
        src_set = src_set[:limit]
    # if dir is a folder of images, load images directly
    else:
        alloc = limit if limit > 0 else 65536
        files = os.listdir(dir)
        np.random.shuffle(files)
        for f in files[:alloc]:
            with Image.open(os.path.join(dir, f)) as img:
                src_set.append(img.copy())
        assert len(src_set) > 0, f"Source directory {dir} is empty"
        if limit > 0 and len(src_set) < alloc:
            print(f"Warning: Source directory {dir} has less than {alloc} images")
    print(f"Loaded {len(src_set)} images")
    if limit == 1:
        print(f"Chosen image: {os.path.join(dir, f)}")
    src_blob = preprocess(
        images=src_set, return_tensors="pt"
    ).pixel_values
    return src_blob


@torch.no_grad()
def interpret_emb(emb):
    nltk.download("brown")
    words = brown.words()
    fdist = nltk.FreqDist(w.lower() for w in words)

    vocab_list = list(map(lambda x: x[0], fdist.most_common(10000)))
    cos_sim = [
        torch.abs(
            torch.nn.functional.cosine_similarity(
                emb, clip_model.encode_text(clip.tokenize(word).to("cuda")), dim=-1
            )
        ).item()
        for word in tqdm(vocab_list)
    ]

    sorted_vocab = sorted(zip(vocab_list, cos_sim), key=lambda x: x[1], reverse=True)
    topk = sorted_vocab[:20]
    print("Top 20 words:")
    for word, sim in topk:
        print(f"\t{word}: {sim}")


@torch.no_grad()
def extract_clip_domain_gap_huggingface(
    src_dir: str,
    tgt_dir: str,
    return_gap_only: bool = True,
    save_loc: Union[str, None] = None,
    mode: str = "mean",
    clip_model_name: str = "ViT-B/32",
    src_data_limit: int = 128,
    tgt_data_limit: int = 32,
    device: str = "cuda",
    interpret: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    clip_processor = CLIPImageProcessor.from_pretrained(
        clip_model_name, subfolder="feature_extractor"
    )
    clip_vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
        clip_model_name, subfolder="image_encoder"
    ).to(device)

    tgt_blob = data_loader_clipProc(tgt_dir, clip_processor, limit=tgt_data_limit).to(
        device
    )
    src_blob = data_loader_clipProc(src_dir, clip_processor, limit=src_data_limit).to(
        device
    )

    src_reps = clip_vision_encoder(src_blob).image_embeds
    tgt_reps = clip_vision_encoder(tgt_blob).image_embeds
    src_mean_rep = torch.mean(src_reps, dim=0)
    tgt_mean_rep = torch.mean(tgt_reps, dim=0)

    if mode == "mean":
        domain_direction = tgt_mean_rep - src_mean_rep
    elif mode == "pca":
        reps_concat = torch.cat([src_reps, tgt_reps], dim=0)
        pca = PCA(n_components=1)
        pca.fit(reps_concat.cpu().numpy())
        domain_direction = torch.from_numpy(pca.components_[0]).to(device)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if interpret:
        # interpret_emb(domain_direction)
        pass

    if save_loc:
        os.makedirs(os.path.dirname(save_loc), exist_ok=True)
        torch.save(domain_direction.cpu(), save_loc)
        print(f"Domain gap direction embeddings saved to {save_loc}")

    if return_gap_only:
        return domain_direction
    else:
        return domain_direction, src_reps, tgt_reps


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir", type=str, help="Path to the source domain directory"
    )
    parser.add_argument(
        "--tgt_dir", type=str, help="Path to the target domain directory"
    )
    parser.add_argument(
        "--src_limit", type=int, default=-1, help="Number of source images to use"
    )
    parser.add_argument(
        "--tgt_limit", type=int, default=-1, help="Number of target images to use"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/domain_reps/domain_gap.pt",
        help="Path to save the domain gap representation",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="mean",
        help="Mode to use for domain gap direction extraction",
    )
    parser.add_argument(
        "--clip_model_name", type=str, default="ViT-B/32", help="CLIP model name"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for CLIP"
    )
    return parser.parse_args()


def main():
    args = get_args()
    extract_clip_domain_gap_huggingface(
        args.src_dir,
        args.tgt_dir,
        save_loc=args.save_path,
        mode=args.mode,
        clip_model_name=args.clip_model_name,
        src_data_limit=args.src_limit,
        tgt_data_limit=args.tgt_limit,
        device=args.device,
        interpret=True,
    )


if __name__ == "__main__":
    main()
