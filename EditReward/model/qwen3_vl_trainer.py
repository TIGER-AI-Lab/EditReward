import torch
import torch.nn as nn
from typing import List, Optional

from transformers import Qwen3VLForConditionalGeneration


class Qwen3_VLRewardModelBT_MultiHead(Qwen3VLForConditionalGeneration):
    """
    Qwen3-VL reward model wrapper.
    - forward(...) returns {"logits": pooled_scores}
    - supports single head (ranknet) and multi-head (ranknet_multi_head / *_regression)
    """

    def __init__(
        self,
        config,
        output_dim=4,
        reward_token="last",
        special_token_ids=None,
        rm_head_type="default",
        rm_head_kwargs=None,
        pooling_strategy="min",
    ):
        super().__init__(config)
        self.output_dim = output_dim
        self.rm_head_type = rm_head_type
        self.reward_token = reward_token
        self.special_token_ids = special_token_ids
        self.pooling_strategy = pooling_strategy

        self.rm_head = None
        self.rm_heads = None

        if rm_head_type == "default":
            self.rm_head = nn.Linear(config.hidden_size, output_dim, bias=False)

        elif rm_head_type == "ranknet" or rm_head_type == "ranknet_share_head":
            if rm_head_kwargs is not None:
                layers = []
                num_layers = rm_head_kwargs.get("num_layers", 3)
                for layer in range(num_layers):
                    if layer == 0:
                        layers = [
                            nn.Linear(config.hidden_size, rm_head_kwargs["hidden_size"]),
                            nn.ReLU(),
                            nn.Dropout(rm_head_kwargs.get("dropout", 0.1)),
                        ]
                    elif layer < num_layers - 1:
                        layers += [
                            nn.Linear(rm_head_kwargs["hidden_size"], rm_head_kwargs["hidden_size"]),
                            nn.ReLU(),
                            nn.Dropout(rm_head_kwargs.get("dropout", 0.1)),
                        ]
                    else:
                        layers.append(
                            nn.Linear(
                                rm_head_kwargs["hidden_size"],
                                output_dim,
                                bias=rm_head_kwargs.get("bias", False),
                            )
                        )
                self.rm_head = nn.Sequential(*layers)
            else:
                self.rm_head = nn.Sequential(
                    nn.Linear(config.hidden_size, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Linear(1024, 16),
                    nn.ReLU(),
                    nn.Linear(16, output_dim),
                )

        elif rm_head_type == "ranknet_multi_head" or rm_head_type == "ranknet_multi_head_regression":
            num_heads = 2
            if rm_head_kwargs is not None:
                num_heads = rm_head_kwargs.get("num_heads", 2)
            self.rm_heads = nn.ModuleDict()
            for i in range(num_heads):
                if rm_head_kwargs is not None:
                    head_layers = []
                    num_layers = rm_head_kwargs.get("num_layers", 3)
                    for layer in range(num_layers):
                        if layer == 0:
                            head_layers += [
                                nn.Linear(config.hidden_size, rm_head_kwargs["hidden_size"]),
                                nn.ReLU(),
                                nn.Dropout(rm_head_kwargs.get("dropout", 0.1)),
                            ]
                        elif layer < num_layers - 1:
                            head_layers += [
                                nn.Linear(rm_head_kwargs["hidden_size"], rm_head_kwargs["hidden_size"]),
                                nn.ReLU(),
                                nn.Dropout(rm_head_kwargs.get("dropout", 0.1)),
                            ]
                        else:
                            head_layers.append(
                                nn.Linear(
                                    rm_head_kwargs["hidden_size"],
                                    output_dim,
                                    bias=rm_head_kwargs.get("bias", False),
                                )
                            )
                    self.rm_heads[f"head_{i}"] = nn.Sequential(*head_layers)
                else:
                    self.rm_heads[f"head_{i}"] = nn.Sequential(
                        nn.Linear(config.hidden_size, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.05),
                        nn.Linear(1024, 16),
                        nn.ReLU(),
                        nn.Linear(16, output_dim),
                    )

        if self.rm_head is not None:
            self.rm_head.to(torch.float32)
        if self.rm_heads is not None:
            for h in self.rm_heads.values():
                h.to(torch.float32)

        if self.special_token_ids is not None:
            self.reward_token = "special"

    def _pool_logits(self, logits, input_ids, batch_size):
        if self.reward_token == "last":
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                )
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            pooled = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
            return pooled

        if self.reward_token == "mean":
            if self.config.pad_token_id is None:
                return logits.mean(dim=1)
            sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)
            valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
            pooled = torch.stack(
                [logits[i, : valid_lengths[i]].mean(dim=0) for i in range(batch_size)]
            )
            return pooled

        if self.reward_token == "special":
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (input_ids == special_token_id)
            pooled = logits[special_token_mask, ...]
            pooled = pooled.view(batch_size, -1)
            return pooled

        raise ValueError("Invalid reward_token")

    def _run_single_batch_through_model_and_head(self, batch_dict, head_module):
        input_ids = batch_dict.get("input_ids", None)
        attention_mask = batch_dict.get("attention_mask", None)
        position_ids = batch_dict.get("position_ids", None)
        past_key_values = batch_dict.get("past_key_values", None)
        inputs_embeds = batch_dict.get("inputs_embeds", None)
        pixel_values = batch_dict.get("pixel_values", None)
        pixel_values_videos = batch_dict.get("pixel_values_videos", None)
        image_grid_thw = batch_dict.get("image_grid_thw", None)
        video_grid_thw = batch_dict.get("video_grid_thw", None)
        return_dict = batch_dict.get("return_dict", True)

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        batch_size = input_ids.shape[0]
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            logits = head_module(hidden_states)
        pooled = self._pool_logits(logits, input_ids, batch_size)
        return pooled

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        batch_dim1: Optional[dict] = None,
        batch_dim2: Optional[dict] = None,
        **kwargs,
    ):
        if self.rm_head_type == "ranknet":
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                return_dict=return_dict if return_dict is not None else True,
            )
            hidden_states = outputs[0]
            batch_size = input_ids.shape[0]
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                logits = self.rm_head(hidden_states)
            pooled = self._pool_logits(logits, input_ids, batch_size)
            return {"logits": pooled}

        if self.rm_head_type == "ranknet_multi_head" or self.rm_head_type == "ranknet_multi_head_regression":
            num_heads = len(self.rm_heads)
            head_batches = [None] * num_heads

            provided_batches = {**kwargs}
            if batch_dim1 is not None:
                provided_batches["batch_dim1"] = batch_dim1
            if batch_dim2 is not None:
                provided_batches["batch_dim2"] = batch_dim2

            for i in range(num_heads):
                key = f"batch_dim{i+1}"
                if key in provided_batches:
                    head_batches[i] = provided_batches[key]

            if all(b is None for b in head_batches):
                raise ValueError(
                    f"No per-head batches found. Expected keys: {[f'batch_dim{j+1}' for j in range(num_heads)]}"
                )

            logits_per_head = []
            for i, b in enumerate(head_batches):
                if b is not None:
                    logits = self._run_single_batch_through_model_and_head(
                        b, self.rm_heads[f"head_{i}"]
                    )
                    logits_per_head.append(logits)

            if self.pooling_strategy is None:
                return logits_per_head

            stacked = torch.stack(logits_per_head, dim=0)
            if self.pooling_strategy == "min":
                final_logits = stacked.min(dim=0).values
            elif self.pooling_strategy == "mean":
                final_logits = stacked.mean(dim=0)
            elif self.pooling_strategy == "sum":
                means = stacked[:, :, 0]
                sigmas = torch.exp(stacked[:, :, 1])
                final_mean = means.sum(dim=0)
                final_var = (sigmas ** 2).sum(dim=0)
                final_sigma = torch.sqrt(final_var)
                final_logits = torch.stack([final_mean, torch.log(final_sigma)], dim=-1)
            else:
                final_logits = stacked.mean(dim=0)

            return {"logits": final_logits}

        if self.rm_head_type == "ranknet_share_head":
            num_heads = 2
            head_batches = [None] * num_heads

            provided_batches = {**kwargs}
            if batch_dim1 is not None:
                provided_batches["batch_dim1"] = batch_dim1
            if batch_dim2 is not None:
                provided_batches["batch_dim2"] = batch_dim2

            for i in range(num_heads):
                key = f"batch_dim{i+1}"
                if key in provided_batches:
                    head_batches[i] = provided_batches[key]

            if all(b is None for b in head_batches):
                raise ValueError(
                    f"No per-head batches found. Expected keys: {[f'batch_dim{j+1}' for j in range(num_heads)]}"
                )

            logits_per_head = []
            for i, b in enumerate(head_batches):
                if b is not None:
                    logits = self._run_single_batch_through_model_and_head(b, self.rm_head)
                    logits_per_head.append(logits)

            if self.pooling_strategy is None:
                return logits_per_head

            stacked = torch.stack(logits_per_head, dim=0)
            if self.pooling_strategy == "min":
                final_logits = stacked.min(dim=0).values
            elif self.pooling_strategy == "mean":
                final_logits = stacked.mean(dim=0)
            else:
                final_logits = stacked.mean(dim=0)

            return {"logits": final_logits}

        raise NotImplementedError(f"rm_head_type {self.rm_head_type} not supported")


