# +
import os
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    WavLMForCTC,
)

# # Base models

class BaseModel(nn.Module):
    """
    BaseFeaturesExtractor class that will extract features according to the type of model
    """

    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self, x):
        outputs = self.model(x)
        return outputs


class Wav2Vec2(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        self.model = Wav2Vec2ForCTC.from_pretrained(params.pretrained_name)
        in_features = self.model.lm_head.in_features
        self.model.lm_head = nn.Linear(
            in_features=in_features, out_features=self.params.vocab_size
        )


class WavLM(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        print("Load WavLM model!!!!!!!")
        self.model = WavLMForCTC.from_pretrained(params.pretrained_name)
        in_features = self.model.lm_head.in_features
        self.model.lm_head = nn.Linear(
            in_features=in_features, out_features=self.params.vocab_size
        )
        print(self.model.lm_head.weight.shape)


# Default parameters
network_param = Namespace(
    network_name="WavLM",
    pretrained_name="microsoft/wavlm-base",  # HuggingFace hub model
    freeze=True,
    freeze_transformer=True,
    eos_token="</s>",
    bos_token="<s>",
    unk_token="<unk>",
    pad_token="<pad>",
    word_delimiter_token="|",
    vocab_size=200,
)

optim_param = Namespace(
    optimizer="AdamW",
    lr=1e-6,
    weight_decay=1e-08,
    accumulate_grad_batches=16,
    use_scheduler=0,
    max_epochs=10,
    warmup_epochs=1,
    warmup_start_lr=1e-7,
    eta_min=1e-07,
    step_size=2,
    gamma=0.1,
    milestones=[8, 10, 15],
    min_lr=5e-09,
    patience=10,
)


# # Lightning Module

class BaseModule(LightningModule):
    def __init__(self, network_param, optim_param, tokenizer=None, total_num_phonemes=None):
        super().__init__()
        self.tokenizer = tokenizer

        if total_num_phonemes is None:
            network_param.vocab_size = tokenizer.total_num_phonemes
        else:
            network_param.vocab_size = total_num_phonemes

        self.optim_param = optim_param
        self.lr = optim_param.lr

        self.loss = nn.CTCLoss(blank=0)

        if network_param.network_name.lower() == "wavlm":
            self.model = WavLM(network_param)
        else:
            self.model = Wav2Vec2(network_param)

        if network_param.freeze:
            self.model.model.freeze_feature_encoder()

        if network_param.freeze_transformer:
            self.model.model.requires_grad_(False)
            self.model.model.lm_head.requires_grad_(True)

        self.set_profiler()
        self.save_hyperparameters()

    def set_profiler(self, profiler=None):
        self.profiler = profiler if profiler is not None else pl.profilers.PassThroughProfiler()

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        loss, logits, preds, targets = self._get_outputs(batch, batch_idx)

        if loss != loss:
            print("loss is nan, model collapse, exiting")
            loss = torch.mean(logits)
            exit(1)

        self.log(
            "train/loss",
            loss,
            batch_size=len(preds),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"loss": loss, "logits": logits.detach(), "preds": preds, "targets": targets}

    def validation_step(self, batch, batch_idx):
        loss, logits, preds, targets = self._get_outputs(batch, batch_idx)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return {"loss": loss, "logits": logits, "preds": preds, "targets": targets}

    def test_step(self, batch, batch_idx):
        loss, logits, preds, targets = self._get_outputs(batch, batch_idx)
        self.log("test/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return {"loss": loss, "logits": logits, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optim_param.optimizer)(
            self.parameters(), lr=self.lr, weight_decay=self.optim_param.weight_decay
        )
        if self.optim_param.use_scheduler:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.optim_param.warmup_epochs,
                max_epochs=self.optim_param.max_epochs,
                warmup_start_lr=self.optim_param.warmup_start_lr,
                eta_min=self.optim_param.eta_min,
            )
            return [[optimizer], [scheduler]]
        return optimizer

    def _get_outputs(self, batch, batch_idx):
        x = batch
        if x["audio"].ndim == 3:
            x["audio"] = x["audio"][:, 0, :]

        with self.profiler.profile("generate phoneme features and logits"):
            y = self.model.model(x["audio"], output_hidden_states=True)
            output = y.logits

        log_probs = F.log_softmax(output, dim=-1)
        log_probs = log_probs.permute(1, 0, 2)

        input_lengths = torch.LongTensor([min(b // 320, output.size(1)) for b in x["audio_length"]])
        targets = x["phoneme_id"]
        target_lengths = x["phoneme_id_length"]

        with self.profiler.profile("generate loss"):
            loss = self.loss(log_probs, targets, input_lengths, target_lengths)

        with self.profiler.profile("generate predict phonemes"):
            pred_phoneme_ids = torch.argmax(output, dim=-1)
            pred_phoneme_ids = [
                pred_phoneme_ids[i, :_len].unique_consecutive() for i, _len in enumerate(input_lengths)
            ]
            phone_preds = self.tokenizer.batch_decode(x["language"], pred_phoneme_ids)

        with self.profiler.profile("combine predict phonemes"):
            labels = [_phoneme_id[: x["phoneme_id_length"][i]] for i, _phoneme_id in enumerate(x["phoneme_id"])]
            phone_targets = self.tokenizer.batch_decode(x["language"], labels)

        return loss, output, phone_preds, phone_targets


def combine_consecutive_identical(strings):
    if not strings:
        return []
    combined = [strings[0]]
    for i in range(1, len(strings)):
        if strings[i] != combined[-1]:
            combined.append(strings[i])
    if combined[0] == "|":
        combined = combined[1:]
    return combined


# ## load model

def load_phoneme_model(network_name="wav2vec", pretrained_path=None, total_num_phonemes=198):
    from ay2.tools.text import Phonemer_Tokenizer_Recombination

    vocab_path = "/workspace/DeepfakeDetectionRenewed/vocab_phoneme"
    print("Now, load vocab json files from", vocab_path, "Please make sure the vocab files are correct")

    languages = ["en", "de", "es", "fr", "it", "pl", "ru", "uk", "zh-CN"]
    tokenizer = Phonemer_Tokenizer_Recombination(
        vocab_files=[os.path.join(vocab_path, f"vocab-phoneme-{language}.json") for language in languages],
        languages=languages,
    )

    network_param.network_name = network_name

    if network_name.lower() == "wavlm":
        network_param.pretrained_name = "microsoft/wavlm-base"
    else:
        network_param.pretrained_name = "facebook/wav2vec2-base-960h"

    if pretrained_path is None:
        model = BaseModule(
            network_param, optim_param, tokenizer=tokenizer, total_num_phonemes=total_num_phonemes
        )
    else:
        model = BaseModule.load_from_checkpoint(
            pretrained_path,
            network_param=network_param,
            optim_param=optim_param,
            tokenizer=tokenizer,
            total_num_phonemes=total_num_phonemes,
            weights_only=False,
        ).cpu()
    return model
