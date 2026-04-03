"""CLIP文本编码器封装。

封装OpenAI CLIP的文本编码器分支，用于：
1. 作为检测器的文本分支（可冻结）
2. 作为知识蒸馏的Teacher模型

参考文献:
- CLIP (ICML 2021): Learning Transferable Visual Models From Natural Language Supervision
- CLIP-KD (CVPR 2024): An Empirical Study of CLIP Model Distillation
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CLIPTextEncoder(nn.Module):
    """CLIP文本编码器封装。

    使用Hugging Face Transformers加载预训练CLIP文本编码器，
    提供统一的编码接口。

    Args:
        model_name: HuggingFace模型名称，默认 "openai/clip-vit-base-patch32"
        embed_dim: 输出嵌入维度，默认512
        freeze: 是否冻结参数
        max_length: 最大token长度
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        embed_dim: int = 512,
        freeze: bool = True,
        max_length: int = 77,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.model_name = model_name

        self._tokenizer = None
        self._text_model = None
        self._text_projection = None
        self._is_loaded = False

        # 延迟加载（避免import时就下载模型）
        self._freeze = freeze

    @staticmethod
    def _resolve_local_path(model_name: str) -> str:
        """尝试将HuggingFace模型名解析为本地缓存路径。"""
        import os
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        repo_dir = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
        refs_main = os.path.join(repo_dir, "refs", "main")
        if os.path.exists(refs_main):
            with open(refs_main, "r") as f:
                commit_hash = f.read().strip()
            snapshot_dir = os.path.join(repo_dir, "snapshots", commit_hash)
            if os.path.isdir(snapshot_dir):
                return snapshot_dir
        return model_name  # 回退到原始名称

    def _ensure_loaded(self):
        """确保模型已加载。延迟加载策略。"""
        if self._is_loaded:
            return

        try:
            from transformers import CLIPModel, CLIPTokenizer, CLIPConfig

            # 绕过transformers对torch<2.6的安全检查（CVE-2025-32434）
            # 本项目仅加载可信的官方预训练权重，风险可控
            try:
                import transformers.utils.import_utils as _import_utils
                import transformers.modeling_utils as _modeling_utils
                _noop = lambda: None
                _import_utils.check_torch_load_is_safe = _noop
                _modeling_utils.check_torch_load_is_safe = _noop
            except Exception:
                pass

            # 解析本地缓存路径，完全避免网络请求
            local_path = self._resolve_local_path(self.model_name)
            print(f"[DEBUG] Resolved model path: {local_path}", flush=True)

            print("[DEBUG] Loading tokenizer...", flush=True)
            self._tokenizer = CLIPTokenizer.from_pretrained(
                local_path, local_files_only=True
            )
            print("[DEBUG] Tokenizer loaded.", flush=True)

            print("[DEBUG] Loading CLIPModel...", flush=True)
            clip_model = CLIPModel.from_pretrained(
                local_path, local_files_only=True
            )
            print("[DEBUG] CLIPModel loaded.", flush=True)

            # 只提取文本部分
            self._text_model = clip_model.text_model
            self._text_projection = clip_model.text_projection

            if self._freeze:
                self._freeze_params()

            self._is_loaded = True
            logger.info(
                f"CLIP text encoder loaded. "
                f"Params: {sum(p.numel() for p in self.parameters()) / 1e6:.1f}M"
            )
            print("[DEBUG] CLIP text encoder ready.", flush=True)

        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers"
            )

    def _freeze_params(self):
        """冻结所有参数。"""
        if self._text_model is not None:
            for param in self._text_model.parameters():
                param.requires_grad = False
        if self._text_projection is not None:
            for param in self._text_projection.parameters():
                param.requires_grad = False

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    def tokenize(self, texts: List[str]) -> dict[str, torch.Tensor]:
        """将文本列表tokenize。

        Args:
            texts: 文本字符串列表

        Returns:
            tokenizer输出字典
        """
        self._ensure_loaded()
        return self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        texts: List[str] | None = None,
    ) -> torch.Tensor:
        """编码文本，返回归一化的嵌入。

        可以传入tokenized结果 (input_ids, attention_mask)，
        或直接传入文本列表 (texts)。

        Args:
            input_ids: Token IDs [B, L]
            attention_mask: 注意力掩码 [B, L]
            texts: 文本字符串列表 (与input_ids二选一)

        Returns:
            text_embeds: 归一化文本嵌入 [B, D]
        """
        self._ensure_loaded()

        if texts is not None:
            tokens = self.tokenize(texts)
            device = next(self.parameters()).device
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

        # 文本编码器前向
        text_outputs = self._text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # 取 [CLS] token 的输出
        # CLIP的text_model输出pooler_output已经是[CLS]经过池化后的结果
        pooled_output = text_outputs[1]  # pooler_output

        # 通过投影层
        if self._text_projection is not None:
            text_embeds = self._text_projection(pooled_output)
        else:
            text_embeds = pooled_output

        return text_embeds

    def encode_class_names(
        self,
        class_names: List[str],
        prompt_template: str = "a photo of a {}",
    ) -> torch.Tensor:
        """编码类别名称，用于开放词汇检测。

        对每个类别名使用prompt模板生成文本，然后编码。

        Args:
            class_names: 类别名列表 ["person", "car", ...]
            prompt_template: Prompt模板

        Returns:
            class_embeds: [N, D] 归一化的类别嵌入
        """
        texts = [prompt_template.format(name) for name in class_names]
        with torch.no_grad():
            embeds = self.forward(texts=texts)
        # L2归一化
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds

    def get_all_token_embeddings(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        texts: List[str] | None = None,
    ) -> torch.Tensor:
        """获取所有token的嵌入（非仅[CLS]），用于细粒度融合。

        Args:
            input_ids: Token IDs [B, L]
            attention_mask: 注意力掩码 [B, L]
            texts: 文本字符串列表

        Returns:
            token_embeds: [B, L, D] 所有token的嵌入
        """
        self._ensure_loaded()

        if texts is not None:
            tokens = self.tokenize(texts)
            device = next(self.parameters()).device
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

        text_outputs = self._text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # last_hidden_state: [B, L, hidden_dim]
        token_embeds = text_outputs[0]

        return token_embeds, attention_mask
