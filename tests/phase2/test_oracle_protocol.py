"""Tests for oracle_protocol.py — OracleProtocol, OracleReply, MockOracle."""
import pytest


def _make_reply(text="ok", satisfied=False, load=0.0):
    from src.oracle_protocol import OracleReply
    return OracleReply(raw_text=text, satisfied=satisfied, turn_cognitive_load=load)


def test_oracle_reply_construction():
    """OracleReply has raw_text, satisfied, turn_cognitive_load fields."""
    from src.oracle_protocol import OracleReply
    reply = OracleReply(raw_text="split cluster 0", satisfied=False, turn_cognitive_load=0.3)
    assert reply.raw_text == "split cluster 0"
    assert reply.satisfied is False
    assert reply.turn_cognitive_load == 0.3


def test_oracle_reply_default_cognitive_load():
    """OracleReply.turn_cognitive_load defaults to 0.0."""
    from src.oracle_protocol import OracleReply
    reply = OracleReply(raw_text="ok", satisfied=True)
    assert reply.turn_cognitive_load == 0.0


def test_mock_oracle_scripted_sequence(tiny_state_3cluster):
    """MockOracle returns scripted replies in order."""
    from src.oracle_protocol import MockOracle
    script = [_make_reply("first"), _make_reply("second"), _make_reply("third")]
    oracle = MockOracle(script=script)
    r0 = oracle.reply(tiny_state_3cluster, "message 0")
    r1 = oracle.reply(tiny_state_3cluster, "message 1")
    r2 = oracle.reply(tiny_state_3cluster, "message 2")
    assert r0.raw_text == "first"
    assert r1.raw_text == "second"
    assert r2.raw_text == "third"


def test_mock_oracle_after_script_exhausted(tiny_state_3cluster):
    """MockOracle returns default neutral reply after script is exhausted."""
    from src.oracle_protocol import MockOracle
    script = [_make_reply("only")]
    oracle = MockOracle(script=script)
    oracle.reply(tiny_state_3cluster, "msg 0")   # consumes script
    fallback = oracle.reply(tiny_state_3cluster, "msg 1")  # beyond script
    assert fallback.satisfied is False
    assert fallback.raw_text == ""


def test_mock_oracle_empty_script_crashes():
    """MockOracle raises AssertionError if script is empty (fail loudly)."""
    from src.oracle_protocol import MockOracle
    with pytest.raises(AssertionError):
        MockOracle(script=[])


def test_mock_oracle_satisfies_oracle_protocol(tiny_state_3cluster):
    """MockOracle is a valid OracleProtocol (structural subtyping check)."""
    from src.oracle_protocol import OracleProtocol, MockOracle
    oracle = MockOracle(script=[_make_reply()])
    assert isinstance(oracle, OracleProtocol)
