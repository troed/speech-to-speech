from __future__ import annotations

from speech_to_speech.pipeline.echo_filter import EchoFilter


class TestEchoFilter:
    def test_empty_is_not_echo(self):
        f = EchoFilter()
        assert not f.is_echo("hello")

    def test_exact_match_is_echo(self):
        f = EchoFilter()
        f.record("I do not have a record of a specific crew member named Poole")
        assert f.is_echo("I do not have a record of a specific crew member named Poole")

    def test_near_match_is_echo(self):
        f = EchoFilter()
        f.record("I do not have a record of a specific crew member named Poole")
        assert f.is_echo("do not have a record of specific crew member Poole")

    def test_different_text_is_not_echo(self):
        f = EchoFilter()
        f.record("I do not have a record of a specific crew member named Poole")
        assert not f.is_echo("What about captain Janeway")

    def test_partial_overlap_below_threshold_is_not_echo(self):
        f = EchoFilter()
        f.record("The weather in Paris is cloudy with a chance of rain this afternoon")
        assert not f.is_echo("What is the weather in Paris")

    def test_empty_record_does_not_crash(self):
        f = EchoFilter()
        f.record("")
        f.record("   ")
        assert not f.is_echo("hello")

    def test_empty_candidate_returns_false(self):
        f = EchoFilter()
        f.record("some text")
        assert not f.is_echo("")
        assert not f.is_echo("   ")

    def test_similar_short_text_is_echo(self):
        f = EchoFilter()
        f.record("Hello there")
        assert f.is_echo("Hello there")

    def test_history_rotation(self):
        f = EchoFilter(max_history=2)
        f.record("first message")
        f.record("second message")
        f.record("third message")
        assert not f.is_echo("first message")
        assert f.is_echo("third message")

    def test_threshold_respected(self):
        f = EchoFilter(threshold=0.9)
        f.record("I do not have a record of a specific crew member named Poole expressing a public stance")
        assert not f.is_echo("What about captain Janeway and the Voyager crew")

    def test_barge_in_not_treated_as_echo(self):
        f = EchoFilter()
        f.record("The capital of France is Paris, a beautiful city known for the Eiffel Tower")
        assert not f.is_echo("What about the weather in Berlin")
