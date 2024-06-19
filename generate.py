from extract_domain_gap import extract_clip_domain_gap_huggingface
import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import time
from typing import Tuple

from diffusers import StableUnCLIPImg2ImgPipeline, ControlNetModel
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from controlnet_aux import HEDdetector
from src.data.driving_data_cvt import gta_to_ade20k, synthia_to_ade20k
from src.models.pipeline_stable_unclip_controlnet_img2img import (
    StableUnCLIPControlNetImg2ImgPipeline,
)

CONTROLNET_MODEL = {
    "canny": "thibaud/controlnet-sd21-canny-diffusers",
    "hed": "thibaud/controlnet-sd21-hed-diffusers",
    "depth": "thibaud/controlnet-sd21-depth-diffusers",
    "seg": "thibaud/controlnet-sd21-ade20k-diffusers",
}
STABLE_DIFFUSION_MODEL = "stabilityai/stable-diffusion-2-1-unclip"
OPENAI_CLIP_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


class DomainGenerator:
    def __init__(
        self,
        save_dir,
        src_dir,
        gap_emb,
        control_config=None,
        lora_loc=None,
        lora_scale=1.0,
        device="cuda",
        seed=42,
    ) -> None:
        self.out_size = 768
        self.clip_size = 224
        if save_dir is not None:
            self.data_save_dir = os.path.join(save_dir, "data")
            self.grid_save_dir = os.path.join(save_dir, "grid")
        self.src_dir = src_dir
        self.device = device
        self.gap_emb = gap_emb.to(torch.float16).to(self.device)
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.processor = CLIPImageProcessor.from_pretrained(
            STABLE_DIFFUSION_MODEL,
            subfolder="feature_extractor",
            torch_dtype=torch.float16,
        )
        self.src_show_transform = transforms.Compose(
            [
                transforms.Resize(self.out_size),
                transforms.CenterCrop((self.out_size, self.out_size)),
            ]
        )
        self.control_img_transform = transforms.Compose(
            [
                transforms.Resize(self.out_size),
                transforms.CenterCrop((self.out_size, self.out_size)),
            ]
        )
        self.tokenizer = None
        self.text_encoder = None
        self.visual_encoder = None
        self.init_visual_clip_models()

        self.controls = []
        if control_config:
            if "canny" in control_config:
                self.controls.append("canny")
                self.canny_lo_thres = control_config["canny"]["lo_thres"]
                self.canny_hi_thres = control_config["canny"]["hi_thres"]
            if "hed" in control_config:
                self.controls.append("hed")
                self.hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
                self.hed = lambda image: self.hed_detector(
                    image, detect_resolution=1024
                )
            if "depth" in control_config:
                self.controls.append("depth")
            if "seg" in control_config:
                self.controls.append("seg")

        self.controlnets = [
            ControlNetModel.from_pretrained(
                CONTROLNET_MODEL[control_type], torch_dtype=torch.float16
            )
            for control_type in self.controls
        ]

        if len(self.controls) > 0:
            self.pipe = StableUnCLIPControlNetImg2ImgPipeline.from_pretrained(
                STABLE_DIFFUSION_MODEL,
                controlnet=self.controlnets,
                torch_dtype=torch.float16,
            ).to(self.device)
        else:
            self.pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
                STABLE_DIFFUSION_MODEL, torch_dtype=torch.float16
            ).to(self.device)

        self.lora_scale = None
        if lora_loc:
            self.pipe.unet.load_attn_procs(lora_loc)
            self.lora_scale = lora_scale
        self.pipe = self.pipe.to(self.device)

        # self.pipe.enable_attention_slicing()
        # self.pipe.enable_vae_slicing()
        # self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()
        torch.cuda.empty_cache()

    def get_src_images(self, n):
        if "gta" in self.src_dir:
            return self.get_gta_images(n)
        elif "synthia" in self.src_dir:
            return self.get_synthia_images(n)
        else:
            return self.get_genric_images(n)

    def get_genric_images(self, n):
        image_file_list = []
        while True:
            if len(image_file_list) < n:
                image_file_list = os.listdir(self.src_dir)
                np.random.shuffle(image_file_list)
            for _ in range(n):
                image_file = image_file_list.pop()
                with Image.open(os.path.join(self.src_dir, image_file)) as img:
                    selection = np.array(img.convert("RGB"))
                proc_images = (
                    self.processor(images=selection, return_tensors="pt")
                    .pixel_values.to(self.device)
                    .to(torch.float16)
                )
                yield image_file, proc_images, selection

    def get_gta_images(self, n):
        image_file_list = []
        while True:
            if len(image_file_list) < n:
                image_file_list = os.listdir(self.src_dir)
                # image_file_list.sort()
                np.random.shuffle(image_file_list)
            for _ in range(n):
                image_file = image_file_list.pop()
                with Image.open(os.path.join(self.src_dir, image_file)) as img:
                    selection = np.array(img.convert("RGB"))
                label_file = os.path.join(
                    self.src_dir.replace("images", "labels"), image_file
                )
                with Image.open(label_file) as img:
                    label = img.copy()
                proc_images = (
                    self.processor(images=selection, return_tensors="pt", padding=True)
                    .pixel_values.to(self.device)
                    .to(torch.float16)
                )
                output = {"img": proc_images, "label": label}
                yield image_file, output, selection

    def get_synthia_images(self, n):
        image_file_list = []
        while True:
            if len(image_file_list) < n:
                image_file_list = os.listdir(self.src_dir)
                image_file_list.sort()
            for _ in range(n):
                image_file = image_file_list.pop()
                with Image.open(os.path.join(self.src_dir, image_file)) as img:
                    selection = np.array(img.convert("RGB"))
                label_file = os.path.join(
                    self.src_dir.replace("RGB", "GT/LABELS"), image_file
                )
                label = cv2.imread(label_file, cv2.IMREAD_UNCHANGED)
                label = label[:, :, 2]
                depth_file = os.path.join(
                    self.src_dir.replace("RGB", "Depth/Depth"), image_file
                )
                with Image.open(depth_file) as img:
                    img = img.convert("L")
                    img_np = np.array(img)
                    img_np = 255 - (img_np / img_np.max() * 255)
                    depth = Image.fromarray(img_np)
                proc_images = (
                    self.processor(images=selection, return_tensors="pt", padding=True)
                    .pixel_values.to(self.device)
                    .to(torch.float16)
                )
                output = {"img": proc_images, "label": label, "depth": depth}
                yield image_file, output, selection

    def init_text_clip_models(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            STABLE_DIFFUSION_MODEL, torch_dtype=torch.float16
        ).to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(
            STABLE_DIFFUSION_MODEL, torch_dtype=torch.float16
        ).to(self.device)

    def init_visual_clip_models(self):
        self.visual_encoder = CLIPVisionModelWithProjection.from_pretrained(
            STABLE_DIFFUSION_MODEL, subfolder="image_encoder", torch_dtype=torch.float16
        ).to(self.device)

    def get_control_image(self, data):
        image = data
        label = None
        depth = None
        if isinstance(data, dict):
            image = data["img"]
            if "label" in data:
                label = data["label"]
            if "depth" in data:
                depth = data["depth"]

        control_images = []
        np_image = image.detach().cpu().numpy()[0]
        np_image = np_image.transpose(1, 2, 0)
        np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min()) * 255
        np_image = np_image.astype(np.uint8)

        for control_type in self.controls:
            if control_type == "canny":
                low_threshold = self.canny_lo_thres
                high_threshold = self.canny_hi_thres

                control_image = cv2.Canny(np_image, low_threshold, high_threshold)
                control_image = control_image[:, :, None]
                control_image = np.concatenate(
                    [control_image, control_image, control_image], axis=2
                )
                control_image = Image.fromarray(control_image)
            elif control_type == "hed":
                control_image = self.hed(np_image)
            elif control_type == "depth":
                if depth is None:
                    raise ValueError("Depth image not provided")
                control_image = self.control_img_transform(depth)
            elif control_type == "seg":
                if label is None:
                    raise ValueError("Label image not provided")
                if "gta" in self.src_dir:
                    control_image = gta_to_ade20k(label)
                    control_image = self.control_img_transform(control_image)
                elif "synthia" in self.src_dir:
                    control_image = synthia_to_ade20k(label)
                    control_image = self.control_img_transform(control_image)
            else:
                raise ValueError(f"Control type {control_type} not supported")
            control_images.append(control_image)
        return control_images

    def generate(
        self,
        edit_weight=1,
        prompt=None,
        neg_prompt=None,
        num_inference_steps=20,
        guidance_scale=10.0,
        controlnet_conditioning_scale=1.0,
        noise_level=0,
        n_per_prompt=4,
        n_batch=4,
        save_grid=False,
    ):
        os.makedirs(self.data_save_dir, exist_ok=True)
        os.makedirs(self.grid_save_dir, exist_ok=True)
        if prompt:
            print(f"Prompt: {prompt} (guidance scale: {guidance_scale})")
        if neg_prompt:
            print(f"Negative Prompt: {neg_prompt}")
        if save_grid:
            print(f"Saving grid images to {self.grid_save_dir}")
        grid_images = []
        data_store = self.get_src_images(n_per_prompt * n_batch)
        for batch in range(n_batch):
            src_filename, src_data, src_img_raw = next(data_store)
            label = None
            if isinstance(src_data, dict):
                src_img = src_data["img"]
            else:
                src_img = src_data
            src_emb = self.visual_encoder(src_img).image_embeds
            guide_emb = src_emb + edit_weight * self.gap_emb
            if save_grid:
                grid_offset = 1
                grid_images.append(
                    self.src_show_transform(Image.fromarray(src_img_raw))
                )

            if len(self.controls) > 0:
                control_images = self.get_control_image(src_data)
                if save_grid:
                    for control_image in control_images:
                        grid_offset += 1
                        grid_images.append(self.src_show_transform(control_image))

                imgs = self.pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=neg_prompt,
                    image_embeds=guide_emb,
                    control_image=control_images,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    num_images_per_prompt=n_per_prompt,
                    cross_attention_kwargs=(
                        {"scale": self.lora_scale} if self.lora_scale else None
                    ),
                    noise_level=noise_level,
                ).images  # type: ignore

            else:
                imgs = self.pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=neg_prompt,
                    image_embeds=guide_emb,
                    num_images_per_prompt=n_per_prompt,
                    cross_attention_kwargs=(
                        {"scale": self.lora_scale} if self.lora_scale else None
                    ),
                    noise_level=noise_level,
                ).images  # type: ignore

            for i, img in enumerate(imgs):
                img.save(
                    os.path.join(self.data_save_dir, f"{batch}_{i}_{src_filename}")
                )
                if save_grid:
                    grid_images.append(img)
                    if len(grid_images) == min(
                        (n_per_prompt + grid_offset) * n_batch,
                        (n_per_prompt + grid_offset) * n_per_prompt,
                    ):
                        width, height = grid_images[0].size
                        grid = Image.new(
                            "RGB",
                            (
                                width * (n_per_prompt + grid_offset),
                                height * min(n_per_prompt, n_batch),
                            ),
                        )
                        for i, img in enumerate(grid_images):
                            grid.paste(
                                img,
                                (
                                    width * (i % (n_per_prompt + grid_offset)),
                                    height * (i // (n_per_prompt + grid_offset)),
                                ),
                            )
                        grid.save(
                            os.path.join(
                                self.grid_save_dir,
                                f"{batch//n_per_prompt}_{src_filename}",
                            )
                        )
                        grid_images = []

    def interpolate(
        self,
        prompt=None,
        neg_prompt=None,
        n_imgs=10,
        n_interpolate=11,
        interpolate_range=1,
    ):
        os.makedirs(self.grid_save_dir, exist_ok=True)
        for img_i in range(n_imgs):
            src_img, src_img_raw = next(self.get_src_images(1))
            src_emb = self.visual_encoder(src_img).image_embeds
            width, height = self.out_size, self.out_size
            grid_img = Image.new("RGB", (width * n_interpolate, height))
            for i, weight in enumerate(
                np.linspace(0, interpolate_range, n_interpolate)
            ):
                if i == 0:
                    src_img_resized = Image.fromarray(src_img_raw[0]).resize(
                        (self.out_size, self.out_size)
                    )
                    grid_img.paste(src_img_resized, (width * i, 0))
                    continue
                guide_emb = src_emb + weight * self.gap_emb
                imgs = self.pipe(
                    prompt=prompt, negative_prompt=neg_prompt, image_embeds=guide_emb
                ).images
                grid_img.paste(imgs[0], (width * i, 0))
            grid_img.save(os.path.join(self.grid_save_dir, f"interpolate_{img_i}.png"))

    def edit_image(self, src_img_raw, prompt=None, neg_prompt=None):
        src_img = (
            self.processor(images=src_img_raw, return_tensors="pt", padding=True)
            .pixel_values.to(self.device)
            .to(torch.float16)
        )
        src_emb = self.visual_encoder(src_img).image_embeds
        guide_emb = src_emb + self.gap_emb
        imgs = self.pipe(
            prompt=prompt, negative_prompt=neg_prompt, image_embeds=guide_emb
        ).images
        return imgs


def get_args():
    parser = argparse.ArgumentParser(
        description="Generation with domain embedding guidance"
    )
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
        "--domain_emb_loc", type=str, help="Path to the precomputed domain embedding"
    )

    parser.add_argument(
        "--gen_src_dir", type=str, help="Path to the generation source directory"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output/sd_emb_addition/",
        help="Path to save the generation results",
    )

    parser.add_argument(
        "--lora_loc", type=str, default=None, help="Path to the finetuned LoRA weights"
    )
    parser.add_argument(
        "--lora_scale", type=float, default=1, help="Scale of the LoRA weights"
    )

    parser.add_argument(
        "--edit_weight", type=float, default=1, help="Weight of the edit"
    )
    parser.add_argument(
        "--n_batch", type=int, default=3, help="Number of batches to generate"
    )
    parser.add_argument(
        "--n_per_prompt", type=int, default=4, help="Number of images per prompt"
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Prompt to further aid the generation"
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default=None,
        help="Prompt to further aid the generation",
    )
    parser.add_argument(
        "--noise_level",
        type=int,
        default=0,
        help="Noise level to add to the generation pipeline",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps to add to the generation pipeline",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=10,
        help="guidance_scale to add to the generation pipeline",
    )

    parser.add_argument(
        "--control_type",
        type=str,
        nargs="*",
        choices=["canny", "hed", "depth", "seg"],
        default=[],
        help="control_type to add to the generation pipeline",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0,
        help="controlnet_conditioning_scale to add to the generation pipeline",
    )
    parser.add_argument(
        "--canny_lo_thres",
        type=int,
        default=100,
        help="canny_lo_thres to add to the generation pipeline",
    )
    parser.add_argument(
        "--canny_hi_thres",
        type=int,
        default=500,
        help="canny_hi_thres to add to the generation pipeline",
    )

    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Whether to interpolate between the source and target domain",
    )
    parser.add_argument(
        "--n_interpolate",
        type=int,
        default=6,
        help="Number of images to interpolate between the source and target domain",
    )
    parser.add_argument(
        "--interpolate_range", type=int, default=1, help="Range of interpolation"
    )

    parser.add_argument(
        "--save_grid", action="store_true", help="Whether to save the grid images"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on"
    )
    return parser.parse_args()


def main():
    args = get_args()
    domain_emb_loc = (
        args.domain_emb_loc
        if args.domain_emb_loc
        else os.path.join(args.save_dir, "domain_gap.pt")
    )
    if os.path.exists(domain_emb_loc):
        domain_gap = torch.load(domain_emb_loc).to(args.device)
    else:
        domain_gap = extract_clip_domain_gap_huggingface(
            args.src_dir,
            args.tgt_dir,
            save_loc=domain_emb_loc,
            clip_model_name=STABLE_DIFFUSION_MODEL,
            src_data_limit=args.src_limit,
            tgt_data_limit=args.tgt_limit,
            device=args.device,
        )

    print("domain gap embedding loaded")
    control_config = {}
    for control_type in args.control_type:
        control_config[control_type] = {}
    if "canny" in args.control_type:
        control_config["canny"] = {
            "lo_thres": args.canny_lo_thres,
            "hi_thres": args.canny_hi_thres,
        }

    generator = DomainGenerator(
        args.save_dir,
        args.gen_src_dir,
        domain_gap,
        control_config=control_config,
        lora_loc=args.lora_loc,
        lora_scale=args.lora_scale,
        device=args.device,
        seed=args.seed,
    )
    if args.interpolate:
        generator.interpolate(
            prompt=args.prompt,
            neg_prompt=args.neg_prompt,
            n_imgs=args.n_batch,
            n_interpolate=args.n_interpolate,
            interpolate_range=args.interpolate_range,
        )
    else:
        generator.generate(
            edit_weight=args.edit_weight,
            prompt=args.prompt,
            neg_prompt=args.neg_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            noise_level=args.noise_level,
            n_per_prompt=args.n_per_prompt,
            n_batch=args.n_batch,
            save_grid=args.save_grid,
        )


if __name__ == "__main__":
    main()