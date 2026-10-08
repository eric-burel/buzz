import json
import os
import itertools
from typing import Callable, List

from buzz.model_loader import _kyutai_model_files
from buzz.transcriber.transcriber import FileTranscriptionTask, Segment, Task


def _timestamped_words(
    text_tokens,
    tokenizer,
    frame_rate: float,
    padding_token_id: int,
    offset_seconds: float,
) -> list[tuple[str, float, float]]:
    """Decode Kyutai's text stream into timestamped word spans."""
    import torch

    text_tokens = text_tokens.cpu().view(-1)
    (boundaries,) = torch.where(text_tokens == 0)
    results = []

    def decode(start: int, end: int) -> None:
        results.extend(
            _decode_timestamped_span(
                text_tokens,
                start,
                end,
                tokenizer,
                frame_rate,
                padding_token_id,
                offset_seconds,
            )
        )

    if boundaries.numel() == 0:
        return results

    boundary_positions = boundaries.tolist()
    for left, right in zip(boundary_positions, boundary_positions[1:]):
        decode(left + 1, right)

    last_start = boundary_positions[-1] + 1
    eos_positions = torch.where(text_tokens[last_start:] == tokenizer.eos_id())[0]
    if eos_positions.numel():
        last_end = last_start + int(eos_positions[0])
    else:
        last_end = min(text_tokens.shape[-1], last_start + int(frame_rate))
    decode(last_start, last_end)
    return results


def _decode_timestamped_span(
    tokens,
    start: int,
    end: int,
    tokenizer,
    frame_rate: float,
    padding_token_id: int,
    offset_seconds: float,
) -> list[tuple[str, float, float]]:
    token_ids = tokens[start:end]
    token_ids = token_ids[token_ids > padding_token_id]
    words = tokenizer.decode(token_ids.tolist()).split()
    if not words:
        return []

    def timestamp(token_start: int, token_end: int) -> tuple[float, float]:
        return (
            max(0.0, token_start / frame_rate - offset_seconds),
            max(0.0, token_end / frame_rate - offset_seconds),
        )

    if len(words) == 1:
        start_time, end_time = timestamp(start, end)
        return [(words[0], start_time, end_time)]

    results = []
    cursor = start
    for word in words[:-1]:
        word_end = cursor + len(tokenizer.encode(word))
        start_time, end_time = timestamp(cursor, word_end)
        results.append((word, start_time, end_time))
        cursor = word_end
    start_time, end_time = timestamp(cursor, end)
    results.append((words[-1], start_time, end_time))
    return results


class _TimestampedWordStream:
    def __init__(
        self,
        tokenizer,
        frame_rate: float,
        padding_token_id: int,
        offset_seconds: float,
        on_word: Callable[[str], None],
    ):
        import torch

        self.torch = torch
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.padding_token_id = padding_token_id
        self.offset_seconds = offset_seconds
        self.on_word = on_word
        self.pending_tokens = torch.empty(0, dtype=torch.long)
        self.pending_start = 0
        self.has_boundary = False

    def update(self, tokens) -> None:
        tokens = tokens.cpu().view(-1)
        if tokens.numel() == 0:
            return
        pending = self.torch.cat((self.pending_tokens, tokens))
        while True:
            (boundaries,) = self.torch.where(pending == 0)
            if boundaries.numel() == 0:
                break
            boundary = int(boundaries[0])
            if self.has_boundary:
                for word, _, _ in _decode_timestamped_span(
                    pending,
                    0,
                    boundary,
                    self.tokenizer,
                    self.frame_rate,
                    self.padding_token_id,
                    self.offset_seconds - self.pending_start / self.frame_rate,
                ):
                    self.on_word(word)
            pending = pending[boundary + 1 :]
            self.pending_start += boundary + 1
            self.has_boundary = True
        self.pending_tokens = pending

    def finish(self) -> None:
        if not self.has_boundary:
            return
        eos_positions = self.torch.where(
            self.pending_tokens == self.tokenizer.eos_id()
        )[0]
        end = (
            int(eos_positions[0])
            if eos_positions.numel()
            else min(
                self.pending_tokens.shape[-1],
                int(self.frame_rate),
            )
        )
        for word, _, _ in _decode_timestamped_span(
            self.pending_tokens,
            0,
            end,
            self.tokenizer,
            self.frame_rate,
            self.padding_token_id,
            self.offset_seconds - self.pending_start / self.frame_rate,
        ):
            self.on_word(word)


class KyutaiTranscriber:
    @staticmethod
    def transcribe(
        task: FileTranscriptionTask,
        on_word: Callable[[str], None] | None = None,
    ) -> List[Segment]:
        if task.transcription_options.task != Task.TRANSCRIBE:
            raise ValueError("Kyutai STT supports transcription only, not translation.")

        repo_id = task.transcription_options.model.hugging_face_model_id
        supported_languages = {
            "kyutai/stt-1b-en_fr": {"en", "fr"},
            "kyutai/stt-2.6b-en": {"en"},
        }.get(repo_id)
        if supported_languages is None:
            raise ValueError(f"Unsupported Kyutai STT model: {repo_id}")
        language = task.transcription_options.language
        if language and language not in supported_languages:
            supported = ", ".join(sorted(supported_languages))
            raise ValueError(
                f"{repo_id} supports {supported}; select a supported language or Detect Language."
            )

        if not task.model_path:
            raise FileNotFoundError(
                "Kyutai STT model is not available locally. Download it from Preferences > Models."
            )
        if not task.file_path:
            raise ValueError("An input media file is required for Kyutai transcription.")
        model_files = _kyutai_model_files(task.model_path)
        if model_files is None or not all(
            os.path.isfile(os.path.join(task.model_path, name))
            for name in model_files
        ):
            raise FileNotFoundError(
                f"Kyutai model files are incomplete in {task.model_path}"
            )

        try:
            import julius
            import moshi.models
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Kyutai support requires the optional dependencies. "
                "Install them with `uv sync --extra kyutai`."
            ) from exc

        import numpy as np

        from buzz import whisper_audio

        config_path = os.path.join(task.model_path, "config.json")
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)

        device = "cpu"
        if torch.cuda.is_available() and os.getenv("BUZZ_FORCE_CPU", "false") == "false":
            device = "cuda"

        info = moshi.models.loaders.CheckpointInfo.from_hf_repo(
            repo_id,
            config_path=config_path,
            moshi_weights=os.path.join(
                task.model_path, config.get("moshi_name", "model.safetensors")
            ),
            mimi_weights=os.path.join(
                task.model_path,
                config.get("mimi_name", "tokenizer-e351c8d8-checkpoint125.safetensors"),
            ),
            tokenizer=os.path.join(
                task.model_path,
                config.get("tokenizer_name", "tokenizer_spm_32k_3.model"),
            ),
        )
        mimi = info.get_mimi(device=device)
        tokenizer = info.get_text_tokenizer()
        language_model = info.get_moshi(device=device, dtype=torch.bfloat16)
        generator = moshi.models.LMGen(language_model, temp=0, temp_text=0.0)

        audio = whisper_audio.load_audio(task.file_path)
        audio_tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        audio_tensor = julius.resample_frac(
            audio_tensor, whisper_audio.SAMPLE_RATE, mimi.sample_rate
        ).to(device)
        if audio_tensor.shape[-1] % mimi.frame_size:
            pad_size = mimi.frame_size - audio_tensor.shape[-1] % mimi.frame_size
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_size))

        audio_delay = info.stt_config.get("audio_delay_seconds", 5.0)
        silence_prefix = info.stt_config.get("audio_silence_prefix_seconds", 1.0)
        padding_token_id = (info.raw_config or {}).get("text_padding_token_id", 3)
        prefix_chunks = int(np.ceil(silence_prefix * mimi.frame_rate))
        suffix_chunks = int(np.ceil(audio_delay * mimi.frame_rate))
        silence = torch.zeros(
            (1, 1, mimi.frame_size), dtype=torch.float32, device=device
        )
        audio_chunks = torch.split(audio_tensor[:, None], mimi.frame_size, dim=-1)
        chunks = itertools.chain(
            itertools.repeat(silence, prefix_chunks),
            audio_chunks,
            itertools.repeat(silence, suffix_chunks),
        )

        generated_tokens = []
        word_stream = (
            _TimestampedWordStream(
                tokenizer,
                mimi.frame_rate,
                padding_token_id,
                prefix_chunks / mimi.frame_rate + audio_delay,
                on_word,
            )
            if on_word is not None
            else None
        )
        with torch.inference_mode(), mimi.streaming(1), generator.streaming(1):
            for audio_chunk in chunks:
                audio_tokens = mimi.encode(audio_chunk)
                text_tokens = generator.step(audio_tokens)
                if text_tokens is not None:
                    generated_tokens.append(text_tokens)
                    if word_stream is not None:
                        word_stream.update(text_tokens)

        if word_stream is not None:
            word_stream.finish()

        if not generated_tokens:
            return []
        all_tokens = torch.cat(generated_tokens, dim=-1)
        words = _timestamped_words(
            all_tokens,
            tokenizer,
            mimi.frame_rate,
            padding_token_id,
            prefix_chunks / mimi.frame_rate + audio_delay,
        )
        if task.transcription_options.word_level_timings:
            return [
                Segment(
                    start=int(start * 1000),
                    end=int(end * 1000),
                    text=text,
                )
                for text, start, end in words
            ]
        if not words:
            return []
        return [
            Segment(
                start=int(words[0][1] * 1000),
                end=int(words[-1][2] * 1000),
                text=" ".join(word for word, _, _ in words),
            )
        ]
