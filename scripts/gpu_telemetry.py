#!/usr/bin/env python3
"""Device telemetry. Polling cadence is not the sensor's measurement bandwidth."""

import csv
import datetime as dt
import math
import statistics
import subprocess
import threading
import time
from pathlib import Path

COLUMNS = ("sample_index", "unix_s", "received_unix_s", "timestamp", "sm_clock_mhz",
           "power_average_w", "temperature_c", "power_instant_w")
MAX_CLOCK_SKEW_S = 1.0


def available(gpu, field):
    result = subprocess.run(["nvidia-smi", "-i", str(gpu), f"--query-gpu={field}",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True)
    try:
        return result.returncode == 0 and math.isfinite(float(result.stdout.strip()))
    except ValueError:
        return False


class ClockSampler:
    def __init__(self, gpu, interval_ms):
        self.gpu, self.interval_ms = str(gpu), int(interval_ms)
        self.samples = []
        self._process = self._reader = None
        if self.interval_ms < 1:
            raise ValueError("telemetry interval must be positive")

    def __enter__(self):
        self.average_field = next((f for f in ("power.draw.average", "power.draw")
                                   if available(self.gpu, f)), None)
        self.instant_field = "power.draw.instant" if available(
            self.gpu, "power.draw.instant") else None
        self.fields = ["timestamp", "clocks.sm", "temperature.gpu"] + [
            f for f in (self.average_field, self.instant_field) if f]
        command = ["nvidia-smi", "-i", self.gpu, f"--query-gpu={','.join(self.fields)}",
                   "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)]
        self._process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        deadline = time.monotonic() + 15
        while not self.samples and time.monotonic() < deadline:
            if self._process.poll() is not None:
                break
            time.sleep(0.005)
        if not self.samples:
            self.__exit__(None, None, None)
            raise RuntimeError("nvidia-smi produced no valid telemetry")
        return self

    def _read(self):
        for line in self._process.stdout:
            received = time.time()
            fields = [v.strip() for v in line.split(",")]
            if len(fields) != len(self.fields):
                continue
            values = dict(zip(self.fields, fields))
            try:
                moment = dt.datetime.strptime(values["timestamp"], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                clock, temperature = float(values["clocks.sm"]), float(values["temperature.gpu"])
                if not (math.isfinite(clock) and clock > 0 and math.isfinite(temperature)):
                    continue
            except ValueError:
                continue
            def power(field):
                try:
                    value = float(values.get(field, ""))
                    return value if math.isfinite(value) and value >= 0 else None
                except ValueError:
                    return None
            self.samples.append((moment, received, values["timestamp"], clock,
                                 power(self.average_field), temperature, power(self.instant_field)))

    def cover_through(self, moment):
        deadline = time.monotonic() + 5
        while (not self.samples or self.samples[-1][0] < moment) and time.monotonic() < deadline:
            if self._process.poll() is not None:
                break
            time.sleep(0.01)
        if not self.samples or self.samples[-1][0] < moment:
            raise RuntimeError("telemetry does not cover the end of the measurement window")

    def __exit__(self, *exception):
        if self._process is None:
            return False
        if self._process.poll() is None:
            self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10)
        self._reader.join(timeout=10)
        self._process.stdout.close()
        return False

    def verify(self):
        if len(self.samples) < 2:
            raise RuntimeError("fewer than two telemetry samples")
        moments = [s[0] for s in self.samples]
        if any(b < a for a, b in zip(moments, moments[1:])):
            raise RuntimeError("telemetry timestamps moved backwards")
        skew = statistics.median(s[1]-s[0] for s in self.samples)
        if abs(skew) > MAX_CLOCK_SKEW_S:
            raise RuntimeError(f"telemetry clock skew {skew:.3f} s")
        return {"sample_count": len(self.samples), "poll_interval_ms": self.interval_ms,
                "power_average_field": self.average_field, "power_instant_field": self.instant_field,
                "power_scope": "GPU board, not CPU+GPU module",
                "sensor_bandwidth": "not inferred from polling cadence", "median_skew_s": skew,
                "first_unix_s": moments[0], "last_unix_s": moments[-1]}

    def write(self, path):
        with Path(path).open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(COLUMNS)
            for index, s in enumerate(self.samples):
                writer.writerow([index, f"{s[0]:.9f}", f"{s[1]:.9f}", s[2], s[3],
                                 "" if s[4] is None else s[4], s[5],
                                 "" if s[6] is None else s[6]])
