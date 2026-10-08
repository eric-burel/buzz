from PyQt6.QtCore import Qt

from buzz.model_loader import KYUTAI_MODEL_IDS, ModelType, TranscriptionModel
from buzz.settings.settings import Settings
from buzz.transcriber.transcriber import Task, TranscriptionOptions
from buzz.widgets.transcriber.transcription_options_group_box import (
    TranscriptionOptionsGroupBox,
)


class TestKyutaiTranscriptionOptions:
    def test_model_selection_limits_task_and_languages(self, qtbot, monkeypatch):
        monkeypatch.setattr(
            Settings,
            "load_custom_model_id",
            lambda _settings, _model: KYUTAI_MODEL_IDS[0],
        )
        monkeypatch.setattr(Settings, "save_custom_model_id", lambda *_args: None)
        options = TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.KYUTAI),
            task=Task.TRANSLATE,
            language="fr",
        )
        widget = TranscriptionOptionsGroupBox(default_transcription_options=options)
        qtbot.add_widget(widget)
        widget.show()

        assert options.task == Task.TRANSCRIBE
        assert not widget.form_layout.isRowVisible(widget.tasks_combo_box)
        assert widget.languages_combo_box.language() == "fr"

        model_combo = widget.whisper_model_size_combo_box
        model_combo.setCurrentIndex(model_combo.findData(KYUTAI_MODEL_IDS[1]))

        assert options.model.hugging_face_model_id == KYUTAI_MODEL_IDS[1]
        assert options.language in (None, "")
        french_index = next(
            index
            for index, (code, _) in enumerate(widget.languages_combo_box.languages)
            if code == "fr"
        )
        assert not (
            widget.languages_combo_box.languages_model.item(french_index).flags()
            & Qt.ItemFlag.ItemIsEnabled
        )
