import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from buzz.model_loader import (
    DOWNLOAD_COMPLETE_MARKER,
    KYUTAI_MODEL_IDS,
    ModelDownloader,
    ModelType,
    TranscriptionModel,
    _kyutai_model_files,
)
from buzz.transcriber.kyutai import (
    KyutaiTranscriber,
    _TimestampedWordStream,
    _timestamped_words,
)
from buzz.transcriber.transcriber import Task


class TestKyutaiModelFiles:
    def test_reads_required_files_from_config(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "moshi_name": "model.safetensors",
                    "mimi_name": "mimi.safetensors",
                    "tokenizer_name": "tokenizer.model",
                }
            )
        )

        assert _kyutai_model_files(str(tmp_path)) == (
            "config.json",
            "model.safetensors",
            "mimi.safetensors",
            "tokenizer.model",
        )

    def test_rejects_config_paths_outside_snapshot(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"moshi_name": "../outside.safetensors"})
        )

        assert _kyutai_model_files(str(tmp_path)) is None


class TestKyutaiModelCache:
    @staticmethod
    def create_snapshot(path):
        (path / "config.json").write_text(
            json.dumps(
                {
                    "moshi_name": "model.safetensors",
                    "mimi_name": "mimi.safetensors",
                    "tokenizer_name": "tokenizer.model",
                }
            )
        )
        for filename in ("model.safetensors", "mimi.safetensors", "tokenizer.model"):
            (path / filename).touch()
        (path / DOWNLOAD_COMPLETE_MARKER).touch()

    def test_only_accepts_complete_cached_snapshots(self, tmp_path):
        model = TranscriptionModel(
            model_type=ModelType.KYUTAI,
            hugging_face_model_id=KYUTAI_MODEL_IDS[0],
        )
        with patch(
            "buzz.model_loader.huggingface_hub.snapshot_download",
            return_value=str(tmp_path),
        ):
            assert model.get_local_model_path() is None
            self.create_snapshot(tmp_path)
            assert model.get_local_model_path() == str(tmp_path)

    def test_downloader_uses_kyutai_file_patterns(self, tmp_path):
        self.create_snapshot(tmp_path)
        model = TranscriptionModel(
            model_type=ModelType.KYUTAI,
            hugging_face_model_id=KYUTAI_MODEL_IDS[0],
        )
        downloader = ModelDownloader(model)
        finished = []
        errors = []
        downloader.signals.finished.connect(finished.append)
        downloader.signals.error.connect(errors.append)

        with patch(
            "buzz.model_loader.download_from_huggingface",
            return_value=str(tmp_path),
        ) as download:
            downloader.run()

        assert finished == [str(tmp_path)]
        assert errors == []
        assert download.call_args.args[0] == KYUTAI_MODEL_IDS[0]
        assert download.call_args.kwargs["allow_patterns"] == [
            "config.json",
            "model.safetensors",
            "mimi-*.safetensors",
            "tokenizer*.model",
        ]


class TestTimestampedWords:
    class Tokenizer:
        def decode(self, tokens):
            return " ".join(
                {4: "hello", 5: "world", 6: "again"}[token] for token in tokens
            )

        def encode(self, word):
            return [word]

        def eos_id(self):
            return 2

    def test_decodes_words_from_delayed_token_stream(self):
        tokens = torch.tensor([[[[0, 4, 5, 0, 6, 2]]]])

        assert _timestamped_words(
            tokens,
            self.Tokenizer(),
            frame_rate=10,
            padding_token_id=3,
            offset_seconds=0,
        ) == [
            ("hello", 0.1, 0.2),
            ("world", 0.2, 0.3),
            ("again", 0.4, 0.5),
        ]

    def test_stream_emits_closed_words_before_the_final_chunk(self):
        words = []
        stream = _TimestampedWordStream(
            self.Tokenizer(),
            frame_rate=10,
            padding_token_id=3,
            offset_seconds=0,
            on_word=words.append,
        )

        stream.update(torch.tensor([0, 4]))
        assert words == []
        stream.update(torch.tensor([5, 0, 6]))
        assert words == ["hello", "world"]
        stream.finish()

        assert words == ["hello", "world", "again"]


class TestKyutaiTranscriberValidation:
    @staticmethod
    def task(model_id, language=None, task=Task.TRANSCRIBE, model_path=""):
        return SimpleNamespace(
            model_path=model_path,
            transcription_options=SimpleNamespace(
                model=SimpleNamespace(hugging_face_model_id=model_id),
                language=language,
                task=task,
            ),
        )

    def test_declares_supported_model_repositories(self):
        assert KYUTAI_MODEL_IDS == (
            "kyutai/stt-1b-en_fr",
            "kyutai/stt-2.6b-en",
        )

    def test_rejects_translation(self):
        with pytest.raises(ValueError, match="transcription only"):
            KyutaiTranscriber.transcribe(
                self.task("kyutai/stt-1b-en_fr", task=Task.TRANSLATE)
            )

    def test_rejects_unsupported_language(self):
        with pytest.raises(ValueError, match="supports en, fr"):
            KyutaiTranscriber.transcribe(
                self.task("kyutai/stt-1b-en_fr", language="de")
            )

    def test_rejects_nonexistent_model_path(self):
        with pytest.raises(FileNotFoundError, match="not available locally"):
            KyutaiTranscriber.transcribe(self.task("kyutai/stt-1b-en_fr"))
