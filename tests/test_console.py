"""Tests for src/console.py."""

import logging

import pytest

from src.console import (ascii_bar, describe_event_es, describe_token_es,
                         format_duration, pitch_to_note_name,
                         pitch_to_note_name_es, section, setup_cli_logging,
                         trend_arrow)
from src.preprocessing import build_vocabulary

_INV_VOCAB = {idx: tok for tok, idx in build_vocabulary().items()}


class TestNoteNames:
    def test_middle_c(self):
        assert pitch_to_note_name(60) == "C4"
        assert pitch_to_note_name_es(60) == "Do4"

    def test_sharp_note(self):
        assert pitch_to_note_name_es(68) == "Sol#4"


class TestDescribeEvents:
    def test_note_on(self):
        assert describe_event_es("NOTE_ON_60") == "Tocar Do4 (C4)"

    def test_note_off(self):
        assert describe_event_es("NOTE_OFF_69") == "Soltar La4 (A4)"

    def test_time_shift(self):
        assert describe_event_es("TIME_SHIFT_5") == "Esperar 50 ms"

    def test_set_velocity(self):
        desc = describe_event_es("SET_VELOCITY_16")
        assert desc.startswith("Intensidad ≈ ")
        assert "de 127" in desc

    def test_unknown_event_passthrough(self):
        assert describe_event_es("FOO") == "FOO"

    def test_describe_token_by_id(self):
        assert describe_token_es(60, _INV_VOCAB) == "Tocar Do4 (C4)"

    def test_describe_unknown_token_id(self):
        assert describe_token_es(9999, {}) == "Token desconocido 9999"


class TestFormatting:
    @pytest.mark.parametrize(
        "seconds, expected",
        [(4.24, "4.2 s"), (192, "3 min 12 s"), (3780, "1 h 03 min"), (-5, "0.0 s")],
    )
    def test_format_duration(self, seconds, expected):
        assert format_duration(seconds) == expected

    def test_ascii_bar_width_and_clamping(self):
        assert ascii_bar(0.5, width=10) == "█████░░░░░"
        assert ascii_bar(2.0, width=4) == "████"
        assert ascii_bar(-1.0, width=4) == "░░░░"

    def test_trend_arrow_loss_and_accuracy(self):
        assert trend_arrow(-0.1, lower_is_better=True) == "↓ mejora"
        assert trend_arrow(0.1, lower_is_better=True) == "↑ empeora"
        assert trend_arrow(0.1, lower_is_better=False) == "↑ mejora"
        assert trend_arrow(0.0) == "→ igual"


class TestLogging:
    def test_section_numbered_banner(self, caplog):
        logger = logging.getLogger("src.test_console")
        with caplog.at_level(logging.INFO, logger="src.test_console"):
            section(logger, "PRUEBA", 2, 4)
        assert "ETAPA 2/4 · PRUEBA" in caplog.text

    def test_setup_cli_logging_prefixes_warnings(self, capsys):
        root = logging.getLogger()
        saved_handlers, saved_level = root.handlers[:], root.level
        try:
            setup_cli_logging()
            logging.getLogger("src.test_console").warning("algo pasó")
            logging.getLogger("src.test_console").info("todo bien")
            err = capsys.readouterr().err
            assert "AVISO: algo pasó" in err
            assert "todo bien" in err
            assert "INFO" not in err
        finally:
            root.handlers[:] = saved_handlers
            root.setLevel(saved_level)
            logging.getLogger("src").setLevel(logging.NOTSET)
