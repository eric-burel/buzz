from PyQt6.QtCore import Qt

from buzz.widgets.transcriber.languages_combo_box import LanguagesComboBox


class TestLanguagesComboBox:
    def test_supported_languages_disable_unsupported_options(self, qtbot):
        widget = LanguagesComboBox(default_language="de")
        qtbot.add_widget(widget)

        widget.set_supported_languages({"", "en", "fr"})

        assert widget.language() == ""
        german_index = next(
            index
            for index, (code, _) in enumerate(widget.languages)
            if code == "de"
        )
        assert not (
            widget.languages_model.item(german_index).flags()
            & Qt.ItemFlag.ItemIsEnabled
        )

    def test_clearing_supported_languages_restores_options(self, qtbot):
        widget = LanguagesComboBox(default_language="en")
        qtbot.add_widget(widget)
        widget.set_supported_languages({"", "en"})

        widget.set_supported_languages(None)

        german_index = next(
            index
            for index, (code, _) in enumerate(widget.languages)
            if code == "de"
        )
        assert (
            widget.languages_model.item(german_index).flags()
            & Qt.ItemFlag.ItemIsEnabled
        )
