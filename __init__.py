from .photodoodle_node import (PhotoDoodleCrop, PhotoDoodleLoader,
                               PhotoDoodleSampler)

# 定义要导出的节点类
NODE_CLASS_MAPPINGS = {

    "PhotoDoodle_Crop": PhotoDoodleCrop,
    "PhotoDoodle_Loader": PhotoDoodleLoader,
    "PhotoDoodle_Sampler": PhotoDoodleSampler
}

# 定义节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoDoodle_Crop": "PhotoDoodle裁切",
    "PhotoDoodle_Loader": "PhotoDoodle加载器",
    "PhotoDoodle_Sampler": "PhotoDoodle采样器",
}

# 导出节点
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 