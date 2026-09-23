#!/usr/bin/env python3
"""Sample SM clock, power and temperature with nvidia-smi during one benchmark."""

import csv
import datetime as dt
import statistics
import subprocess
import threading
import time
from pathlib import Path

QUERY_FIELDS = ("timestamp", "clocks.sm", "power.draw", "temperature.gpu")
COLUMNS = ("sample_index", "unix_s", "received_unix_s", "timestamp",
           "sm_clock_mhz", "power_w", "temperature_c")
MAX_CLOCK_SKEW_S = 1.0


class ClockSampler:
    """Stream nvidia-smi samples on a background thread while a benchmark runs.

    nvidia-smi stamps every sample with the host wall clock, which is the clock the
    benchmark also reports for each timed launch, so samples can be attributed to a
    configuration afterwards without perturbing the measured kernels.
    """

    def __init__(self, gpu, interval_ms):
        self.gpu = str(gpu)
        self.interval_ms = int(interval_ms)
        self.samples = []
        self._process = None
        self._reader = None

    def __enter__(self):
        command = ["nvidia-smi", "-i", self.gpu, f"--query-gpu={','.join(QUERY_FIELDS)}",
                   "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)]
        self._process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        # Sampling must already be running when the first timed launch starts.
        deadline = time.monotonic() + 15
        while not self.samples and time.monotonic() < deadline:
            time.sleep(0.005)
        if not self.samples:
            self.__exit__(None, None, None)
            raise RuntimeError("nvidia-smi produced no clock samples")
        return self

    def _read(self):
        for line in self._process.stdout:
            received = time.time()
            fields = [value.strip() for value in line.split(",")]
            if len(fields) != len(QUERY_FIELDS):
                continue
            try:
                moment = dt.datetime.strptime(fields[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                values = [float(field) for field in fields[1:]]
            except ValueError:
                continue
            self.samples.append((moment, received, fields[0], *values))

    def __exit__(self, *exception):
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10)
        # The reader drains the pipe to EOF before the handle is released.
        self._reader.join(timeout=10)
        self._process.stdout.close()
        return False

    def verify(self):
        """Reject samples whose own timestamps disagree with this process's clock."""
        if len(self.samples) < 2:
            raise RuntimeError("fewer than two clock samples were collected")
        skew = statistics.median(sample[1] - sample[0] for sample in self.samples)
        if abs(skew) > MAX_CLOCK_SKEW_S:
            raise RuntimeError(f"nvidia-smi timestamps are skewed by {skew:.3f} s")
        return {"sample_count": len(self.samples), "interval_ms": self.interval_ms,
                "median_skew_s": skew, "first_unix_s": self.samples[0][0],
                "last_unix_s": self.samples[-1][0]}

    def write(self, path):
        with Path(path).open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(COLUMNS)
            for index, sample in enumerate(self.samples):
                writer.writerow([index, f"{sample[0]:.6f}", f"{sample[1]:.6f}", sample[2],
                                 f"{sample[3]:.1f}", f"{sample[4]:.2f}", f"{sample[5]:.1f}"])
