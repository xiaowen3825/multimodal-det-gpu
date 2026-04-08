"""文本处理工具。

- 类别名到prompt模板转换
- 同义词增强
- Tokenizer封装
"""

from __future__ import annotations

from typing import Dict, List

# COCO类别的prompt模板
PROMPT_TEMPLATES = [
    "a photo of a {}",
    "a {}",
    "a picture of a {}",
    "an image of a {}",
    "there is a {} in the image",
]

# 部分COCO类别的同义词 (用于训练时增强)
SYNONYMS: Dict[str, List[str]] = {
    "person": ["person", "man", "woman", "people", "human"],
    "car": ["car", "vehicle", "automobile"],
    "dog": ["dog", "puppy", "canine"],
    "cat": ["cat", "kitten", "feline"],
    "bicycle": ["bicycle", "bike", "cycle"],
    "motorcycle": ["motorcycle", "motorbike"],
    "airplane": ["airplane", "plane", "aircraft"],
    "bus": ["bus", "coach"],
    "tv": ["tv", "television", "monitor", "screen"],
    "laptop": ["laptop", "notebook computer"],
    "cell phone": ["cell phone", "mobile phone", "smartphone"],
    "couch": ["couch", "sofa"],
}


def make_prompt(class_name: str, template: str = "a photo of a {}") -> str:
    """生成文本prompt。

    Args:
        class_name: 类别名
        template: Prompt模板

    Returns:
        prompt字符串
    """
    return template.format(class_name)


def make_prompts_batch(
    class_names: List[str],
    template: str = "a photo of a {}",
    use_synonyms: bool = False,
) -> List[str]:
    """批量生成文本prompt。

    Args:
        class_names: 类别名列表
        template: Prompt模板
        use_synonyms: 是否使用同义词增强

    Returns:
        prompt字符串列表
    """
    prompts = []
    for name in class_names:
        if use_synonyms and name in SYNONYMS:
            import random
            synonym = random.choice(SYNONYMS[name])
            prompts.append(template.format(synonym))
        else:
            prompts.append(template.format(name))
    return prompts


def ensemble_prompts(class_names: List[str]) -> Dict[str, List[str]]:
    """生成多模板prompt集合 (用于推理时的prompt ensemble)。

    Args:
        class_names: 类别名列表

    Returns:
        {class_name: [prompt1, prompt2, ...]}
    """
    result = {}
    for name in class_names:
        result[name] = [t.format(name) for t in PROMPT_TEMPLATES]
    return result
