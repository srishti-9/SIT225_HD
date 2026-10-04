"""Serial capture for the DHT22 + PIR Arduino, with automatic reconnect.

Expected Arduino output, one line per reading:  temperature,humidity,occupancy
A failed DHT22 read is sent as:                 NaN,NaN,ERROR
"""
import csv
import os
import sys
import time
from datetime import datetime

import serial


def _ts():
    return time.strftime("%H:%M:%S")


def open_serial_with_retry(port, baud, max_retries=5,
                           backoff_s=2, max_backoff_s=30):
    """Open the serial port, retrying with exponential backoff."""
    attempt = 0
    while True:
        try:
            ser = serial.Serial(port, baud, timeout=2)
            time.sleep(2)  # allow the Arduino to reset
            print(f"{_ts()} Serial connected on {port}")
            return ser
        except (serial.SerialException, OSError) as exc:
            attempt += 1
            print(f"{_ts()} Serial open failed (attempt {attempt}): {exc}")
            if max_retries and attempt >= max_retries:
                raise
            time.sleep(min(backoff_s * 2 ** (attempt - 1), max_backoff_s))


class ResilientSerial:
    """Serial wrapper that transparently reconnects if the link drops."""

    def __init__(self, port, baud, max_retries=5, backoff_s=2):
        self.port, self.baud = port, baud
        self.max_retries, self.backoff_s = max_retries, backoff_s
        self.reconnects = 0
        self.ser = open_serial_with_retry(port, baud, max_retries, backoff_s)

    def read_line(self):
        """Return a stripped line, or None if nothing was read or the
        link was re-established. The caller skips None and loops again."""
        try:
            return self.ser.readline().decode(errors="ignore").strip()
        except (serial.SerialException, OSError):
            print(f"{_ts()} Serial link dropped, attempting reconnect...")
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = open_serial_with_retry(
                self.port, self.baud, self.max_retries, self.backoff_s)
            self.reconnects += 1
            return None

    def close(self):
        self.ser.close()


def main(port, baud=9600, out_path="raw_data.csv"):
    new_file = not os.path.exists(out_path)
    link = ResilientSerial(port, baud)
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "temperature", "humidity", "occupancy"])
        try:
            while True:
                line = link.read_line()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != 3:
                    continue
                w.writerow([datetime.now().isoformat(timespec="seconds")] + parts)
                f.flush()
        except KeyboardInterrupt:
            pass
        finally:
            link.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "COM3")
