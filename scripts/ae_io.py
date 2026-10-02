"""
WFS (Acoustic Emission) file reader.

Refactored and translated to Python 3 from https://github.com/jove1/ae,
focused on the functionality for .wfs files.
"""
from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from struct import calcsize, unpack_from
from typing import Iterator, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class PrettyOrderedDict(OrderedDict):
    """OrderedDict with a readable __str__."""

    def __str__(self, prefix: str = "") -> str:
        indent = "    "
        lines = ["OrderedDict("]
        for k, v in self.items():
            if isinstance(v, PrettyOrderedDict):
                lines.append(f"{prefix + indent}({k!r}, {v.__str__(prefix + indent)}),")
            else:
                lines.append(f"{prefix + indent}({k!r}, {v!r}),")
        lines.append(prefix + ")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data / WFS classes
# ---------------------------------------------------------------------------

class Data:
    """Base class providing iteration over raw blocks of AE data."""

    progress = staticmethod(lambda pct, elapsed: None)

    # subclasses must set these -----------------------------------------------
    fname: str
    block_dtype: np.dtype
    timescale: float
    datascale: list[float]
    channels: int
    shape: tuple
    size: int
    dtype: np.dtype

    # ------------------------------------------------------------------ utils
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.fname!r})"

    def calc_sizes(self, file_size: int) -> None:
        n_blocks = file_size // self.block_dtype.itemsize
        rest = file_size % self.block_dtype.itemsize
        if rest:
            warnings.warn(f"{rest} bytes at the end of file won't fit into blocks")

        tmp = self._get_block_data(np.empty(0, self.block_dtype))
        self.dtype = tmp.dtype
        self.shape = (n_blocks,) + tmp.shape[1:-1]
        self.size = int(np.prod(self.shape))
        self.channels = tmp.shape[-1]
        assert self.channels == len(self.datascale)

    # -------------------------------------------------------- block iteration
    def iter_blocks(
        self,
        start: int = 0,
        stop: float = float("inf"),
        channel: int | slice = slice(None),
    ) -> Iterator[Tuple[int, np.ndarray]]:
        progress = self.progress
        import time
        start_time = time.time()
        for pos, raw in self.raw_iter_blocks(start, stop):
            self._check_block(pos, raw)
            yield pos, self._get_block_data(raw)[..., channel]
            progress(100.0 * pos / self.size, time.time() - start_time)
        progress(100, time.time() - start_time)

    # ---------------------------------------------------- subclass overrides
    @staticmethod
    def _get_block_data(d: np.ndarray) -> np.ndarray:
        return d

    def _check_block(self, pos: int, raw: np.ndarray) -> None:
        pass

    def raw_iter_blocks(
        self, start: int = 0, stop: float = float("inf")
    ) -> Iterator[Tuple[int, np.ndarray]]:
        raise NotImplementedError


class ContiguousBase(Data):
    """Base for files where blocks are laid out contiguously in the file."""

    @staticmethod
    def _get_block_data(d: np.ndarray) -> np.ndarray:
        return d

    def _check_block(self, pos: int, raw: np.ndarray) -> None:
        pass

    def raw_iter_blocks(
        self, start: int = 0, stop: float = float("inf")
    ) -> Iterator[Tuple[int, np.ndarray]]:
        buffer = np.empty(8 * 1024 * 1024 // self.block_dtype.itemsize, self.block_dtype)
        block_size = self._get_block_data(buffer)[0, ..., 0].size

        pos = start // block_size * block_size
        seek = start // block_size * buffer.itemsize
        with open(self.fname, "rb", buffering=0) as fh:
            fh.seek(self._offset + seek)
            while True:
                read = fh.readinto(buffer)
                if read < buffer.size * buffer.itemsize:
                    break
                yield pos, buffer
                pos += buffer.size * block_size
                if pos > stop:
                    return

        remains = read // buffer.itemsize
        rest = read % buffer.itemsize
        if remains:
            yield pos, buffer[:remains]
        if rest:
            self._check_rest(buffer.view("B")[read - rest : read])

    def _check_rest(self, data: np.ndarray) -> None:
        warnings.warn(f"{data.size} bytes left in the buffer")


class WFS(ContiguousBase):
    """Reader for .wfs (Acoustic Emission) binary data files."""

    CH_BLOCK_DTYPE = np.dtype([
        ("size", "u2"),
        ("id1", "u1"),
        ("id2", "u1"),
        ("unknown1", "S6"),
        ("chan", "u1"),
        ("zeros", "S7"),
        ("x", "u4"),
        ("unknown2", "S4"),
        ("y", "u4"),
        ("data", "i2", (1024,)),
    ])

    def __init__(self, fname: str, *, checks: bool = False, unknown_meta: bool = False) -> None:
        self.fname = fname
        self.checks = checks

        with open(self.fname, "rb") as fh:
            file_size = os.fstat(fh.fileno()).st_size
            self._offset = self._parse_meta(fh.read(1024), unknown_meta=unknown_meta)

        # determine number of channels
        buf = np.empty(10, self.CH_BLOCK_DTYPE)
        with open(self.fname, "rb", buffering=0) as fh:
            fh.seek(self._offset)
            fh.readinto(buf)

        nch = 1
        for ch in range(1, 10):
            if buf["chan"][ch] == ch + 1:
                nch = ch + 1
            elif buf["chan"][ch] == 1:
                break
            else:
                raise ValueError(f"Invalid channels: {buf['chan']}")

        self.meta["start_time"] = buf["y"][0] * 0.002 / self.meta["hwsetup"]["rate"]

        self.block_dtype = np.dtype([("ch_data", self.CH_BLOCK_DTYPE, (nch,))])
        self.datascale = [self.meta["hwsetup"]["max.volt"] / 32768.0] * nch
        self.timescale = 0.001 / self.meta["hwsetup"]["rate"]
        self.timeunit = "s"
        self.dataunit = "V"

        self.calc_sizes(file_size - self._offset)

    # ------------------------------------------------------------- meta parse
    def _parse_meta(self, data: bytes, unknown_meta: bool = False) -> int:
        self.meta = PrettyOrderedDict()
        offset = 0
        while offset < len(data):
            size, id1, id2 = unpack_from("<HBB", data, offset)
            if (size, id1, id2) == (2076, 174, 1):
                return offset
            offset += 2
            if id1 in (173, 174):
                offset += 2
                size -= 2
                if (id1, id2) == (174, 42):
                    fmt = [
                        ("ver", "H"), ("AD", "B"), ("num", "H"), ("size", "H"),
                        ("id", "B"), ("unk1", "H"), ("rate", "H"),
                        ("trig.mode", "H"), ("trig.src", "H"), ("trig.delay", "h"),
                        ("unk2", "H"), ("max.volt", "H"), ("trig.thresh", "H"),
                    ]
                    sfmt = "<" + "".join(code for _, code in fmt)
                    assert calcsize(sfmt) == size
                    self.meta["hwsetup"] = PrettyOrderedDict(
                        zip([n for n, _ in fmt], unpack_from(sfmt, data, offset))
                    )
                    if self.meta["hwsetup"]["AD"] == 2:
                        self.meta["hwsetup"]["AD"] = "16-bit signed"
                elif unknown_meta:
                    self.meta[(id1, id2)] = data[offset : offset + size]
            else:
                offset += 1
                size -= 1
                if id1 == 99:
                    self.meta["date"] = data[offset : offset + size].rstrip(b"\0\n").decode("utf-8")
                elif id1 == 41:
                    self.meta["product"] = PrettyOrderedDict([
                        ("ver", unpack_from("<xH", data, offset)[0]),
                        ("text", data[offset + 3 : offset + size].rstrip(b"\r\n\0\x1a").decode("utf-8")),
                    ])
                elif unknown_meta:
                    self.meta[id1] = data[offset : offset + size]
            offset += size
        raise ValueError("Data block not found")

    # ---------------------------------------------------------- block helpers
    @staticmethod
    def _get_block_data(d: np.ndarray) -> np.ndarray:
        return d["ch_data"]["data"].swapaxes(-1, -2)

    def _check_block(self, pos: int, raw: np.ndarray) -> None:
        if self.checks:
            assert np.all(raw["size"] == 2076)
            assert np.all(raw["id1"] == 174)
            assert np.all(raw["id2"] == 1)

    def _check_rest(self, data: np.ndarray) -> None:
        if not np.all(data == (7, 0, 15, 255, 255, 255, 255, 255, 127)):
            warnings.warn(f"{data.size} bytes left in the buffer")


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def get_wfs_data(wfs_file_path: str) -> np.ndarray:
    """Read the acoustic stream from a .wfs data file.

    Parameters
    ----------
    wfs_file_path : str
        Path to the .wfs file.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 1 + n_channels)`` where column 0 is the sample
        time (seconds) and the remaining columns are the channel amplitudes
        (Volts).
    """
    wfs = WFS(wfs_file_path)
    T = wfs.size
    dt = wfs.timescale
    dU = wfs.datascale[0]
    nch = wfs.channels

    time_arr = np.arange(T) * dt

    signal_channels: list[np.ndarray] = []
    for ch in range(nch):
        sig = np.concatenate([d.flatten() for _, d in wfs.iter_blocks(start=0, channel=ch)])
        sig = sig.astype(np.float64) * dU
        signal_channels.append(sig)

    data = np.column_stack((time_arr, *signal_channels))
    return data
