from typing import Optional
import os

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import QComboBox, QWidget, QFrame
from PyQt6.QtGui import QStandardItem, QStandardItemModel

from buzz.locale import _
from buzz.transcriber.transcriber import LANGUAGES


class LanguagesComboBox(QComboBox):
    """LanguagesComboBox displays a list of languages available to use with Whisper"""

    # language is a language key from whisper.tokenizer.LANGUAGES or '' for "detect language"
    languageChanged = pyqtSignal(str)

    def __init__(
        self, default_language: Optional[str], parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)

        favorite_languages = os.getenv("BUZZ_FAVORITE_LANGUAGES", '')
        favorite_languages = favorite_languages.split(",")
        favorite_languages = [(lang, LANGUAGES[lang].title()) for lang in favorite_languages
                              if lang in LANGUAGES]
        if favorite_languages:
            favorite_languages.insert(0, ("-------", "-------"))
            favorite_languages.append(("-------", "-------"))

        whisper_languages = sorted(
            [(lang, LANGUAGES[lang].title()) for lang in LANGUAGES],
            key=lambda lang: lang[1],
        )
        self.languages = [("", _("Detect Language"))] + favorite_languages + whisper_languages
        self._unfiltered_languages = list(self.languages)

        self.languages_model = QStandardItemModel()
        for lang in self.languages:
            item = QStandardItem(lang[1])
            if lang[0] == "-------":
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEnabled)
            self.languages_model.appendRow(item)

        self.setModel(self.languages_model)
        self.currentIndexChanged.connect(self.on_index_changed)

        default_language_key = default_language if default_language != "" else None
        for i, lang in enumerate(self.languages):
            if lang[0] == default_language_key:
                self.setCurrentIndex(i)

    def set_supported_languages(self, language_codes: Optional[set[str]]) -> None:
        """Disable languages not supported by the current transcription model."""
        current_language = self.language()
        self.languages = list(self._unfiltered_languages)
        for index, language in enumerate(self.languages):
            code = language[0]
            item = self.languages_model.item(index)
            if item is None or code == "-------":
                continue
            flags = item.flags()
            if language_codes is None or code in language_codes:
                item.setFlags(flags | Qt.ItemFlag.ItemIsEnabled)
            else:
                item.setFlags(flags & ~Qt.ItemFlag.ItemIsEnabled)
        selected_language = (
            current_language
            if language_codes is None or current_language in language_codes
            else ("" if "" in language_codes else next(iter(language_codes)))
        )
        for index, language in enumerate(self.languages):
            code = language[0]
            if code == selected_language:
                self.setCurrentIndex(index)
                return

    def language(self) -> str:
        index = self.currentIndex()
        return self.languages[index][0] if index >= 0 else ""

    def on_index_changed(self, index: int):
        self.languageChanged.emit(self.languages[index][0])

    def showPopup(self):
        super().showPopup()
        popup = self.findChild(QFrame)
        if popup and popup.height() > 400:
            popup.setFixedHeight(400)
