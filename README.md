# ComfyUI PhotoDoodle SVDQuant

用于在comfyUI中使用nunchaku项目的svdquant量化技术去调用PhotoDoodle项目。


1. 原仓库: [PhotoDoodle](https://github.com/showlab/photodoodle) 
2. [nunchaku](https://github.com/mit-han-lab/nunchaku/tree/main)
3. [ComfyUI_PhotoDoodle](https://github.com/smthemex/ComfyUI_PhotoDoodle)
4. [ComfyUI-SC-PhotoDoodle](https://github.com/latentcat/ComfyUI-SC-PhotoDoodle/tree/main)

感谢各位作者大大的分享。


## 说明
- 由于原项目修改flux的前向传播代码，无法直接在comfyui中使用，ComfyUI-SC-PhotoDoodle项目实现了增强版的 Flux 模型，具体可以参考prestartup_script.py下的DoodleFlux类
- 项目代码是在nunchaku项目和photodoodle项目上基础上开发的，个人只是做了简单封装
- 运行该项目需要至少16GB显存，否则会OOM，建议开启cpu_offload

## 安装说明

1. 克隆项目
```powershell
cd custom_nodes
git clone https://github.com/singlinhaha/Comfyui_Heygem_Docker.git
```

2. 安装依赖

nunchaku项目的依赖请参考[nunchaku](https://github.com/mit-han-lab/nunchaku/tree/main)

3. 模型下载

int4的量化模型

量化模型放在models/diffusion_models下
```powershell
huggingface-cli download mit-han-lab/svdq-int4-flux.1-dev --local-dir models/diffusion_models/svdq-int4-flux.1-dev
```

photodoodle的lora模型

放在models/lora下

| Lora name | Function | Trigger word |
|-----------|----------|--------------|
| [sksmonstercalledlulu](https://huggingface.co/nicolaus-huang/PhotoDoodle/blob/main/sksmonstercalledlulu.safetensors) | PhotoDoodle model trained on Cartoon monster dataset | by sksmonstercalledlulu |
| [sksmagiceffects](https://huggingface.co/nicolaus-huang/PhotoDoodle/blob/main/sksmagiceffects.safetensors) | PhotoDoodle model trained on 3D effects dataset | by sksmagiceffects |
| [skspaintingeffects](https://huggingface.co/nicolaus-huang/PhotoDoodle/blob/main/skspaintingeffects.safetensors) | PhotoDoodle model trained on Flowing color blocks dataset | by skspaintingeffects |
| [sksedgeeffect](https://huggingface.co/nicolaus-huang/PhotoDoodle/blob/main/sksedgeeffect.safetensors) | PhotoDoodle model trained on Hand-drawn outline dataset | by sksedgeeffect |

运行节点时还会Hugging Face自动下载black-forest-labs/FLUX.1-dev模型，放在默认缓存目录下

ubuntu: /home/user/.cache/huggingface
windows: C:\Users\username\.cache\huggingface