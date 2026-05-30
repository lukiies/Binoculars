from typing import Union

import os
import gc
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import assert_tokenizer_consistency
from .metrics import perplexity, entropy

torch.set_grad_enabled(False)

huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

# selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1


def _build_quant_config(quantization: Union[str, None], compute_dtype: torch.dtype):
    """Return a BitsAndBytesConfig for the requested quantization mode, or None.

    quantization:
        None / "none" / "full" -> no quantization (full precision / bf16)
        "8bit"                  -> LLM.int8() 8-bit weights
        "4bit"                  -> NF4 4-bit weights w/ double quant (bf16 compute)
    """
    if quantization in (None, "none", "full"):
        return None
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    raise ValueError(f"Invalid quantization mode: {quantization!r}")


class Binoculars(object):
    """Binoculars zero-shot LLM-text detector.

    Extended for 8 GB single-GPU use:
      * ``quantization`` ("8bit" | "4bit" | None) loads each model with
        bitsandbytes so a 7B model fits in VRAM.
      * ``sequential=True`` keeps only ONE model on the GPU at a time
        (load observer -> get logits -> free -> load performer -> get logits).
        This is what lets two 7B models be scored on an 8 GB card, since the
        metrics only need the two *logits* tensors (a few MB), never both
        *models* resident at once.
    """

    def __init__(self,
                 observer_name_or_path: str = "tiiuae/falcon-7b",
                 performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 quantization: Union[str, None] = None,
                 sequential: bool = False,
                 trust_remote_code: bool = False,
                 micro_batch_size: int = 1,
                 ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.change_mode(mode)
        self.observer_name_or_path = observer_name_or_path
        self.performer_name_or_path = performer_name_or_path
        self.quantization = quantization
        self.sequential = sequential
        self.trust_remote_code = trust_remote_code
        # In sequential mode, process at most this many chunks per forward pass
        # so only a small slice of the context is resident on the GPU at once
        # (keeps peak VRAM well under 8 GB on an RTX 4060).
        self.micro_batch_size = max(1, int(micro_batch_size))
        self.compute_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        self.device = DEVICE_1

        # In sequential mode models are (un)loaded on demand inside _get_logits.
        self.observer_model = None
        self.performer_model = None
        if not self.sequential:
            self.observer_model = self._load_model(observer_name_or_path, DEVICE_1)
            self.performer_model = self._load_model(performer_name_or_path, DEVICE_2)
            self.observer_model.eval()
            self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def _load_model(self, name_or_path: str, device: str):
        quant_config = _build_quant_config(self.quantization, self.compute_dtype)
        kwargs = dict(
            trust_remote_code=self.trust_remote_code,
            token=huggingface_config["TOKEN"],
        )
        if quant_config is not None:
            # bitsandbytes requires the model placed via device_map on the GPU.
            kwargs["quantization_config"] = quant_config
            kwargs["device_map"] = {"": device}
        else:
            kwargs["device_map"] = {"": device}
            kwargs["dtype"] = self.compute_dtype
        model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
        model.eval()
        return model

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding):
        # Resident (non-sequential) path: both models already live on the GPU.
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    @torch.inference_mode()
    def _forward_all(self, model_name_or_path: str, encodings_list: list):
        """Load ONE model, run every micro-batch, offload logits to CPU, free model.

        Only one 7B model + one micro-batch of activations is ever GPU-resident,
        which is what keeps peak VRAM under 8 GB.
        """
        model = self._load_model(model_name_or_path, DEVICE_1)
        logits_list = []
        for enc in encodings_list:
            out = model(**enc.to(DEVICE_1)).logits
            if DEVICE_1 != "cpu":
                torch.cuda.synchronize()
            logits_list.append(out.detach().to("cpu"))
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Free THIS model before returning so the next one never co-resides with
        # it. The reference must be dropped in this scope: passing `model` to a
        # helper and `del`-ing the argument there does NOT release it (the caller
        # still holds a reference), and PyTorch modules have reference cycles, so
        # we collect cycles and only then empty the CUDA cache.
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return logits_list

    def _compute_sequential(self, batch: list) -> list:
        # Group chunks into micro-batches (padded within a group only).
        groups = [batch[i:i + self.micro_batch_size]
                  for i in range(0, len(batch), self.micro_batch_size)]
        encodings_list = [self._tokenize(g) for g in groups]

        observer_logits = self._forward_all(self.observer_name_or_path, encodings_list)
        performer_logits = self._forward_all(self.performer_name_or_path, encodings_list)

        scores = []
        for enc, obs, perf in zip(encodings_list, observer_logits, performer_logits):
            enc = enc.to(DEVICE_1)
            obs = obs.to(DEVICE_1)
            perf = perf.to(DEVICE_1)
            ppl = perplexity(enc, perf)
            x_ppl = entropy(obs, perf, enc, self.tokenizer.pad_token_id)
            scores.extend((ppl / x_ppl).tolist())
            del obs, perf
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return scores

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        single = isinstance(input_text, str)
        batch = [input_text] if single else list(input_text)

        if self.sequential:
            binoculars_scores = self._compute_sequential(batch)
        else:
            encodings = self._tokenize(batch)
            observer_logits, performer_logits = self._get_logits(encodings)
            ppl = perplexity(encodings, performer_logits)
            x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                            encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
            binoculars_scores = (ppl / x_ppl).tolist()

        return binoculars_scores[0] if single else binoculars_scores

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(binoculars_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated"
                        ).tolist()
        return pred
