"""Mocked-hardware tests for the serial reconnect logic (3 tests)."""
import pytest
import serial

import capture_serial as cs


class FakeSerial:
    def __init__(self, lines):
        self.lines, self.closed = list(lines), False

    def readline(self):
        item = self.lines.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(cs.time, "sleep", lambda s: None)


def test_reconnect_after_drop(monkeypatch):
    first = FakeSerial([b"20.1,50.0,1\n", serial.SerialException("unplugged")])
    second = FakeSerial([b"20.2,50.1,0\n"])
    ports = iter([first, second])
    monkeypatch.setattr(cs.serial, "Serial", lambda *a, **k: next(ports))
    link = cs.ResilientSerial("COM3", 9600)
    assert link.read_line() == "20.1,50.0,1"
    assert link.read_line() is None
    assert first.closed and link.reconnects == 1
    assert link.read_line() == "20.2,50.1,0"


def test_gives_up_after_max_retries(monkeypatch):
    def always_fail(*a, **k):
        raise serial.SerialException("no device")
    monkeypatch.setattr(cs.serial, "Serial", always_fail)
    with pytest.raises(serial.SerialException):
        cs.open_serial_with_retry("COM3", 9600, max_retries=3)


def test_oserror_also_handled(monkeypatch):
    first = FakeSerial([OSError("device disconnected")])
    second = FakeSerial([b"21.0,49.0,1\n"])
    ports = iter([first, second])
    monkeypatch.setattr(cs.serial, "Serial", lambda *a, **k: next(ports))
    link = cs.ResilientSerial("COM3", 9600)
    assert link.read_line() is None
    assert link.read_line() == "21.0,49.0,1"
