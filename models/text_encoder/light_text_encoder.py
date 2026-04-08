"""轻量文本编码器：4层Transformer，通过知识蒸馏从CLIP学习。

设计目标：
- 参数量: ~12M (vs CLIP ViT-B/32 Text: ~63M)
- 推理速度: 比CLIP快3-4倍
- 与CLIP共享tokenizer

参考文献:
- CLIP-KD (CVPR 2024): An Empirical Study of CLIP Model Distillation
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightTextEncoder(nn.Module):
    """轻量级文本编码器。

    4层Transformer结构，通过知识蒸馏从CLIP文本编码器学习。
    共享CLIP的tokenizer，保持接口一致。

    Args:
        vocab_size: 词汇表大小 (CLIP tokenizer: 49408)
        embed_dim: 输出嵌入维度
        num_layers: Transformer层数
        num_heads: 注意力头数
        hidden_dim: FFN隐藏维度
        max_length: 最大token长度
        dropout: Dropout比率
        pretrained: 预训练权重路径
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        max_length: int = 77,
        dropout: float = 0.1,
        pretrained: str = "",
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.vocab_size = vocab_size

        # Token嵌入
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # 位置嵌入 (可学习)
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_length, embed_dim) * 0.02
        )

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN (更稳定)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # 输出投影
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # LayerNorm
        self.ln_final = nn.LayerNorm(embed_dim)

        self._init_weights()

        # 加载预训练权重
        if pretrained:
            self._load_pretrained(pretrained)

    def _init_weights(self):
        """初始化权重。"""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.01)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _load_pretrained(self, path: str):
        """加载蒸馏训练后的预训练权重。"""
        import logging
        logger = logging.getLogger(__name__)
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.load_state_dict(state_dict, strict=False)
            logger.info(f"Light text encoder weights loaded from: {path}")
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        texts: list[str] | None = None,
    ) -> torch.Tensor:
        """编码文本。

        Args:
            input_ids: Token IDs [B, L]
            attention_mask: 注意力掩码 [B, L]
            texts: 文本字符串列表（需要先加载tokenizer）

        Returns:
            text_embeds: [B, D] 文本嵌入
        """
        if texts is not None:
            input_ids, attention_mask = self._tokenize(texts)
            device = self.token_embedding.weight.device
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

        b, seq_len = input_ids.shape

        # Token + Position Embedding
        x = self.token_embedding(input_ids) + self.position_embedding[:, :seq_len, :]

        # 创建因果掩码 (CLIP风格的单向注意力)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device),
            diagonal=1,
        )

        # padding掩码
        if attention_mask is not None:
            # True表示需要屏蔽
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None

        # Transformer前向
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )

        x = self.ln_final(x)

        # 取EOS token位置的输出 (CLIP风格)
        # EOS是序列中最大的非padding token位置
        if attention_mask is not None:
            eos_indices = attention_mask.sum(dim=1) - 1  # [B]
        else:
            eos_indices = torch.full((b,), seq_len - 1, device=x.device, dtype=torch.long)

        pooled = x[torch.arange(b, device=x.device), eos_indices]

        # 输出投影
        text_embeds = self.output_proj(pooled)

        return text_embeds

    def get_all_token_embeddings(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        texts: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """获取所有token的嵌入。

        Args:
            input_ids: Token IDs [B, L]
            attention_mask: 注意力掩码 [B, L]
            texts: 文本字符串列表

        Returns:
            (token_embeds, attention_mask): ([B, L, D], [B, L])
        """
        if texts is not None:
            input_ids, attention_mask = self._tokenize(texts)
            device = self.token_embedding.weight.device
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

        b, seq_len = input_ids.shape

        x = self.token_embedding(input_ids) + self.position_embedding[:, :seq_len, :]

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device),
            diagonal=1,
        )

        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None

        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )

        x = self.ln_final(x)

        return x, attention_mask

    def encode_class_names(
        self,
        class_names: list[str],
        prompt_template: str = "a photo of a {}",
    ) -> torch.Tensor:
        """编码类别名称列表。

        Args:
            class_names: 类别名列表
            prompt_template: Prompt模板

        Returns:
            class_embeds: [N, D] 归一化的类别嵌入
        """
        texts = [prompt_template.format(name) for name in class_names]
        with torch.no_grad():
            embeds = self.forward(texts=texts)
        embeds = F.normalize(embeds, dim=-1)
        return embeds

    def _tokenize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """使用CLIP tokenizer进行tokenize。"""
        try:
            from transformers import CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            tokens = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return tokens["input_ids"], tokens["attention_mask"]
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
