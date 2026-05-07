# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FunAudioChat Multimodal Plugins - Full Implementation.

Extracted from original OmniChat_v2 with audio processing capabilities.
"""

import inspect
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch
from typing_extensions import override

logger = logging.getLogger(__name__)

ESTIMATED_AUDIO_TOKEN_FPS = 25

if TYPE_CHECKING:
    from llamafactory.data.mm_plugin import ImageInput, VideoInput, AudioInput, MMProcessor

# Import constants from llamafactory
try:
    from llamafactory.extras.constants import AUDIO_PLACEHOLDER
    from llamafactory.data.mm_plugin import BasePlugin
except ImportError as e:
    logger.error(f"Failed to import from llamafactory: {e}")
    raise


def load_audio_batch_default(paths: list[str], sampling_rate: int) -> list[np.ndarray]:
    """
    Default audio loader using librosa.
    
    Args:
        paths: List of audio file paths
        sampling_rate: Target sampling rate
        
    Returns:
        List of audio arrays
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required for audio loading. Install with: pip install librosa")
    
    audios = []
    for path in paths:
        if not path:
            audios.append(np.array([]))
            continue
        try:
            audio, sr = librosa.load(path, sr=sampling_rate, mono=True)
            audios.append(audio)
        except Exception as e:
            logger.warning(f"Failed to load audio from {path}: {e}")
            audios.append(np.array([]))
    
    return audios


def _is_array_like_audio(audio: object) -> bool:
    return isinstance(audio, np.ndarray) or (
        isinstance(audio, list) and len(audio) > 0 and isinstance(audio[0], (int, float, np.integer, np.floating))
    )


def _estimate_audio_token_string(audio: Union[str, np.ndarray, list[float]], processor: "MMProcessor") -> str:
    sampling_rate = getattr(processor, "audio_sampling_rate", 16000) or 16000
    audio_pad_token = getattr(processor, "audio_pad_token", "<|audio_pad|>")

    if _is_array_like_audio(audio):
        audio_array = np.asarray(audio)
        duration_seconds = float(audio_array.shape[0]) / float(sampling_rate)
    else:
        audio_path = str(audio)
        try:
            import soundfile as sf

            info = sf.info(audio_path)
            if not info.frames or not info.samplerate:
                raise RuntimeError("Invalid audio metadata.")
            duration_seconds = float(info.frames) / float(info.samplerate)
        except Exception:
            try:
                import librosa
            except ImportError as exc:
                raise ImportError("librosa is required for audio duration estimation.") from exc

            duration_seconds = float(librosa.get_duration(path=audio_path))

    num_audio_tokens = max(1, int(duration_seconds * ESTIMATED_AUDIO_TOKEN_FPS))
    return audio_pad_token * num_audio_tokens


@dataclass
class FunAudioChatPlugin(BasePlugin):
    """
    FunAudioChat multimodal plugin for audio, image and video processing.
    """
    audio_reader: str = "read_audio"  # Can be "read_audio" or custom reader

    @staticmethod
    def load_audio_batch(
        paths: list[str], sampling_rate: int, audio_reader: str = "read_audio"
    ) -> list[np.ndarray]:
        """
        Loads a batch of audio files.
        
        Args:
            paths: List of audio file paths
            sampling_rate: Target sampling rate
            audio_reader: Audio reader type (default: "read_audio")
            
        Returns:
            List of audio arrays
        """
        return load_audio_batch_default(paths, sampling_rate)

    def _parse_audio_input(self, audio: "AudioInput", processor: "MMProcessor") -> dict[str, object]:
        if _is_array_like_audio(audio):
            audio_array = np.asarray(audio, dtype=np.float32)
            return {
                "path": "",
                "wav_path": "",
                "text": "",
                "token": _estimate_audio_token_string(audio_array, processor),
                "_audio_source": audio_array,
                "_has_feature": True,
            }

        if isinstance(audio, dict):
            parsed_audio = dict(audio)
        elif isinstance(audio, str):
            try:
                parsed_audio = json.loads(audio)
            except json.JSONDecodeError:
                parsed_audio = {"path": audio, "text": ""}
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio)}")

        if not isinstance(parsed_audio, dict):
            raise TypeError(f"Expected audio payload to decode to dict, got: {type(parsed_audio)}")

        audio_path = parsed_audio.get("path", parsed_audio.get("wav_path", "")) or ""
        parsed_audio.setdefault("path", audio_path)
        parsed_audio.setdefault("wav_path", audio_path)
        parsed_audio.setdefault("text", "")

        if not parsed_audio.get("token"):
            if not audio_path:
                raise ValueError("Audio input is missing both `token` and a usable audio path.")
            parsed_audio["token"] = _estimate_audio_token_string(audio_path, processor)

        parsed_audio["_audio_source"] = audio_path
        parsed_audio["_has_feature"] = bool(audio_path)
        return parsed_audio

    def _parse_audios(
        self, audios: list["AudioInput"], processor: Optional["MMProcessor"]
    ) -> list[dict[str, object]]:
        if processor is None:
            raise ValueError("Processor is required to parse audio inputs.")

        return [self._parse_audio_input(audio, processor) for audio in audios]

    def _collect_audio_sources(
        self, parsed_audios: list[dict[str, object]], processor: "MMProcessor"
    ) -> list["AudioInput"]:
        audio_sources = [audio_data["_audio_source"] for audio_data in parsed_audios if audio_data["_has_feature"]]
        if not audio_sources:
            return []

        if any(isinstance(source, np.ndarray) for source in audio_sources):
            normalized_sources: list["AudioInput"] = []
            sampling_rate = getattr(processor, "audio_sampling_rate", 16000)
            for source in audio_sources:
                if isinstance(source, np.ndarray):
                    normalized_sources.append(source)
                else:
                    normalized_sources.extend(
                        FunAudioChatPlugin.load_audio_batch(
                            [str(source)],
                            sampling_rate=sampling_rate,
                            audio_reader=self.audio_reader,
                        )
                    )

            return normalized_sources

        return [str(source) for source in audio_sources]

    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
        imglens: Optional[list[int]] = None,
    ) -> dict[str, "torch.Tensor"]:
        """Process multimodal inputs including audio."""
        
        # Load audio files if they are paths
        if len(audios) > 0 and (isinstance(audios[0], str) or isinstance(audios[0], list)):
            audios = FunAudioChatPlugin.load_audio_batch(
                audios, 
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                audio_reader=self.audio_reader
            )

        mm_inputs = {}
        
        # Process audios
        if len(audios) != 0:
            feature_extractor = getattr(processor, "feature_extractor", None)
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            
            with torch.inference_mode():
                mm_inputs.update(
                    feature_extractor(
                        audios,
                        sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                        return_attention_mask=True,
                        padding=True,
                        return_tensors="pt",
                    )
                )
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask")
            
            # Fix length mismatch between input_features and feature_attention_mask
            input_features = mm_inputs["input_features"]
            feature_attention_mask = mm_inputs["feature_attention_mask"]
            
            input_seq_len = input_features.shape[-1]  # [batch, feature_dim, seq_len]
            mask_seq_len = feature_attention_mask.shape[-1]  # [batch, seq_len]
            
            if input_seq_len != mask_seq_len:
                min_seq_len = min(input_seq_len, mask_seq_len)
                input_features = input_features[..., :min_seq_len]
                feature_attention_mask = feature_attention_mask[..., :min_seq_len]
                mm_inputs["input_features"] = input_features
                mm_inputs["feature_attention_mask"] = feature_attention_mask
                
        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        """Process messages and replace audio placeholders."""
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        
        bos_token: str = getattr(processor, "audio_bos_token")
        eos_token: str = getattr(processor, "audio_eos_token")
        messages = deepcopy(messages)
        parsed_audios = self._parse_audios(audios, processor)

        TEMP_REPLACEMENT_MARKER = "<--[INTERNAL_AUDIO_MARKER]-->" 
        
        if self.expand_mm_tokens:
            audio_tokens = [str(audio_data["token"]) for audio_data in parsed_audios]
            _audio_inputs = processor.speech_tokenizer(
                audio_tokens, 
                return_attention_mask=True, 
                return_token_type_ids=False, 
                padding=True, 
                return_tensors="pt"
            )
            speech_lengths = _audio_inputs["attention_mask"].sum(-1).tolist()

        for message in messages:
            content = message["content"]
            replace_str = []
            
            while AUDIO_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    audio_seqlen = speech_lengths.pop(0)
                    audio_seqlen = (audio_seqlen + (processor.audio_group_size - 1)) // processor.audio_group_size
                else:
                    audio_seqlen = 1

                replace_str.append(f"{bos_token}{self.audio_token * int(audio_seqlen)}{eos_token}")
                content = content.replace(AUDIO_PLACEHOLDER, TEMP_REPLACEMENT_MARKER, 1)

            while TEMP_REPLACEMENT_MARKER in content:
                content = content.replace(TEMP_REPLACEMENT_MARKER, replace_str.pop(0), 1)
            
            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        """Get multimodal inputs for training."""
        mm_inputs = {}

        if audios is not None:
            parsed_audios = self._parse_audios(audios, processor)

            # Filter assistant wav based on <|audio_bos|>
            for i in range(len(parsed_audios)):
                if str(parsed_audios[i]["token"]).startswith('<|audio_bos|>') and str(parsed_audios[i]["token"]).count('AU') > 0:
                    if "wav_path" in parsed_audios[i]:
                        parsed_audios[i]["wav_path"] = ""
                    if "path" in parsed_audios[i]:
                        parsed_audios[i]["path"] = ""
                    parsed_audios[i]["_audio_source"] = ""
                    parsed_audios[i]["_has_feature"] = False

            audio_sources = self._collect_audio_sources(parsed_audios, processor)
            audio_tokens = [str(_audio_data["token"]) for _audio_data in parsed_audios]
            audio_texts = [str(_audio_data.get("text", "")) for _audio_data in parsed_audios]

            audio_inputs = self._get_mm_inputs(images, videos, audio_sources, processor)
            audio_inputs['feature_exist_mask'] = torch.tensor(
                [bool(_audio_data["_has_feature"]) for _audio_data in parsed_audios], 
                dtype=torch.bool
            )

            if audios is not None and len(audios) != 0:
                _audio_inputs = processor.speech_tokenizer(
                    audio_tokens, 
                    return_attention_mask=True, 
                    return_token_type_ids=False, 
                    padding=True, 
                    return_tensors="pt"
                )
                audio_inputs["speech_ids"] = _audio_inputs.pop("input_ids")
                audio_inputs["speech_attention_mask"] = _audio_inputs.pop("attention_mask")

            if audio_texts is not None and len(audio_texts) != 0:
                audio_inputs['text_ids'] = processor.tokenizer(
                    audio_texts, 
                    return_attention_mask=False, 
                    return_token_type_ids=False, 
                    padding=True, 
                    return_tensors="pt"
                )['input_ids']
                audio_inputs['text_attention_mask'] = torch.tensor(
                    [tid != '' for tid in audio_texts], 
                    dtype=torch.bool, 
                    device=audio_inputs['text_ids'].device
                )

            mm_inputs.update(audio_inputs)

        return mm_inputs


__all__ = [
    "FunAudioChatPlugin",
]
