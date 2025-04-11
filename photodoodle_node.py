import os
import torch
import numpy as np
from PIL import Image
import folder_paths

from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
from nunchaku.lora.flux.compose import compose_lora
from nunchaku.caching.diffusers_adapters.flux import apply_cache_on_transformer
from nunchaku.utils import is_turing, load_state_dict_in_safetensors
from .src.pipeline_pe_clone import FluxPipeline
from .node_utils import tensor2pil, pil2tensor

MAX_SEED = np.iinfo(np.int32).max
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

class PhotoDoodleCrop:
    """
    图片裁切节点：在保持原图比例的情况下，尽可能最大化地裁切出目标宽高的区域
    如果原图尺寸不足，则放大至目标尺寸，保持比例且不留白
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_image"
    CATEGORY = "PhotoDoodle"

    def crop_image(self, image, width, height):
        """
        裁切图片，保持比例，最大化填充目标区域，无留白

        参数:
            image: 输入图片张量 [B, H, W, C]
            width: 目标宽度
            height: 目标高度

        返回:
            裁切后的图片张量 [B, height, width, C]
        """
        # 转换为numpy处理
        result = []
        for img in image:
            img_np = img.cpu().numpy()

            # 获取原始尺寸
            orig_h, orig_w = img_np.shape[0], img_np.shape[1]

            # 计算目标比例和原始比例
            target_ratio = width / height
            orig_ratio = orig_w / orig_h

            # 策略：先调整比例（缩小或裁切），再缩放到目标尺寸
            if orig_ratio > target_ratio:
                # 原图更宽，需要调整宽度
                new_w = int(orig_h * target_ratio)
                offset_w = (orig_w - new_w) // 2
                adjusted = img_np[:, offset_w : offset_w + new_w, :]
            else:
                # 原图更高，需要调整高度
                new_h = int(orig_w / target_ratio)
                offset_h = (orig_h - new_h) // 2
                adjusted = img_np[offset_h : offset_h + new_h, :]

            # 调整后的图像缩放到目标尺寸
            pil_img = Image.fromarray((adjusted * 255).astype(np.uint8))
            resized_pil = pil_img.resize((width, height), Image.LANCZOS)
            resized_np = np.array(resized_pil).astype(np.float32) / 255.0

            result.append(torch.from_numpy(resized_np))

        # 堆叠所有处理后的图片
        return (torch.stack(result),)


class PhotoDoodleLoader:
    """
    模型加载节点：加载PhotoDoodle管道
    """
    @classmethod
    def INPUT_TYPES(s):
        prefixes = folder_paths.folder_names_and_paths["diffusion_models"][0]
        local_folders = set()
        for prefix in prefixes:
            if os.path.exists(prefix) and os.path.isdir(prefix):
                local_folders_ = os.listdir(prefix)
                local_folders_ = [
                    folder
                    for folder in local_folders_
                    if not folder.startswith(".") and os.path.isdir(os.path.join(prefix, folder))
                ]
                local_folders.update(local_folders_)
        model_paths = sorted(list(local_folders))

        # 判断gpu
        all_turing = True
        for i in range(torch.cuda.device_count()):
            if not is_turing(f"cuda:{i}"):
                all_turing = False

        if all_turing:
            attention_options = ["nunchaku-fp16"]  # turing GPUs do not support flashattn2
            dtype_options = ["float16"]
        else:
            attention_options = ["nunchaku-fp16", "flash-attention2"]
            dtype_options = ["bfloat16", "float16"]

        return {
            "required": {
                "svdquant_model": (["none"] + model_paths,),
                "pre_lora": (["none"] + [i for i in folder_paths.get_filename_list("loras") if "pre" in i],),
                "loras": (["none"] + folder_paths.get_filename_list("loras"),),
                "attention": (
                    attention_options,
                    {
                        "default": attention_options[0],
                        "tooltip": "Attention implementation. The default implementation is `flash-attention2`. "
                                   "`nunchaku-fp16` use FP16 attention, offering ~1.2× speedup. "
                                   "Note that 20-series GPUs can only use `nunchaku-fp16`.",
                    },
                ),
                "cpu_offload": (
                    ["auto", "enable", "disable"],
                    {
                        "default": "auto",
                        "tooltip": "Whether to enable CPU offload for the transformer model. 'auto' will enable it if the GPU memory is less than 14G.",
                    },
                ),
                "data_type": (
                    dtype_options,
                    {
                        "default": dtype_options[0],
                        "tooltip": "Specifies the model's data type. Default is `bfloat16`. "
                                   "For 20-series GPUs, which do not support `bfloat16`, use `float16` instead.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL_PhotoDoodle",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "PhotoDoodle"

    def loader_main(self, svdquant_model, pre_lora, loras,
                    attention, cpu_offload,
                    data_type):
        prefixes = folder_paths.folder_names_and_paths["diffusion_models"][0]
        for prefix in prefixes:
            if os.path.exists(os.path.join(prefix, svdquant_model)):
                svdquant_model = os.path.join(prefix, svdquant_model)
                break

        # Get the GPU properties
        device_id = 0
        gpu_properties = torch.cuda.get_device_properties(device_id)
        gpu_memory = gpu_properties.total_memory / (1024**2)  # Convert to MiB
        gpu_name = gpu_properties.name
        print(f"GPU {device_id} ({gpu_name}) Memory: {gpu_memory} MiB")

        # Check if CPU offload needs to be enabled
        if cpu_offload == "auto":
            if gpu_memory < 14336:  # 14GB threshold
                cpu_offload_enabled = True
                print("VRAM < 14GiB，enable CPU offload")
            else:
                cpu_offload_enabled = False
                print("VRAM > 14GiB，disable CPU offload")
        elif cpu_offload == "enable":
            cpu_offload_enabled = True
            print("Enable CPU offload")
        else:
            cpu_offload_enabled = False
            print("Disable CPU offload")

        # 加载transformer模型
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            svdquant_model,
            offload=cpu_offload_enabled,
            device=device,
            torch_dtype=torch.float16 if data_type == "float16" else torch.bfloat16,
        )
        if attention == "nunchaku-fp16":
            transformer.set_attention_impl("nunchaku-fp16")
        else:
            assert attention == "flash-attention2"
            transformer.set_attention_impl("flashattn2")


        if pre_lora != "none":
            pre_lora_path = folder_paths.get_full_path("loras", pre_lora)
        else:
            raise ValueError("No model selected")

        if loras != "none":
            lora_path = folder_paths.get_full_path("loras", loras)
        else:
            raise ValueError("No model selected")
        composed_lora = compose_lora(
            [
                (pre_lora_path, 1),
                (lora_path, 1),
            ]
        )
        transformer.update_lora_params(composed_lora)

        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
        ).to(device)

        return (pipeline, )


class PhotoDoodleSampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("MODEL_PhotoDoodle",),
                "image": ("IMAGE",),
                "prompt": (
                "STRING", {"default": "add a halo and wings for the cat by sksmagiceffects", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED, "step": 1, "display": "number"}),
                "width": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 64, "display": "number"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1024, "step": 1, "display": "number"}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "max_sequence_length": (
                "INT", {"default": 512, "min": 128, "max": 512, "step": 1, "display": "number"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sampler_main"
    CATEGORY = "PhotoDoodle"

    def sampler_main(self, pipeline, image, prompt, seed, width, height, steps, guidance_scale, max_sequence_length,
                     **kwargs):
        condition_image = tensor2pil(image)

        result = pipeline(
            prompt=prompt,
            condition_image=condition_image,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            max_sequence_length=max_sequence_length,
            generator = torch.Generator(device=device).manual_seed(seed)
        ).images[0]

        result = pil2tensor(result)
        return (result, )