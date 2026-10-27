"""Fast reader for SAM 3.1 video output.

The official meta-sam-parser decodes a mask in pure Python, about 0.2 s per
mask, so all OpenGraph episodes would take many hours. This module reads the
same raw text and runs the same range decoder compiled with numba. It was
checked mask by mask against meta-sam-parser 0.1.0 (scripts/sam_fast.py --check).

    from sam_fast import read_frames
    for frame, objs in read_frames(path):   # path to *_events.jsonl(.gz)
        for o in objs:
            o.id, o.box, o.mask()           # mask() is a bool array in box space, placed at o.x0, o.y0

Raw line format: <Nf> then, per object,
id<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|><|mask;x=..;y=..;data=H,W,~payload|>
"""

import gzip
import json
import re
from pathlib import Path

import numba as nb
import numpy as np

_EXCLUDED = set('"\\,;<>|')
_ALPHABET = "".join(chr(c) for c in range(0x21, 0x7F) if chr(c) not in _EXCLUDED)
_VALUES = np.full(128, -1, np.int64)
for _i, _c in enumerate(_ALPHABET):
    _VALUES[ord(_c)] = _i

M32 = 0xFFFFFFFF
TOP, BOTTOM = 1 << 24, 1 << 16

_FRAME = re.compile(r"<(\d+)f>")
_OBJ = re.compile(r"(\d+)<\|box;x1=([\d.]+);y1=([\d.]+);x2=([\d.]+);y2=([\d.]+);w=\d+;h=\d+\|>"
                  r"(?:<\|mask;x=(\d+);y=(\d+);data=(\d+),(\d+),([^|]+)\|>)?")


@nb.njit(cache=True)
def _unpack(src, values):
    length = 0
    for i in range(5):
        length = length * 85 + values[src[i]]
    rem = length % 4
    out = np.zeros(length, np.uint8)
    s, d, full = 5, 0, length - rem
    while d < full:
        v = 0
        for i in range(5):
            v = v * 85 + values[src[s + i]]
        s += 5
        out[d] = (v >> 24) & 255
        out[d + 1] = (v >> 16) & 255
        out[d + 2] = (v >> 8) & 255
        out[d + 3] = v & 255
        d += 4
    if rem > 0:
        v = 0
        for i in range(5):
            v = v * 85 + (values[src[s + i]] if i < rem + 1 else 84)
        for i in range(rem):
            out[d + i] = (v >> (24 - i * 8)) & 255
    return out


@nb.njit(cache=True)
def _decode_lossless(packed, width, height, idx, pool):
    """Port of meta_sam_parser._mask_codec._decode_lossless_raster. Returns coverage 0..255."""
    selector = packed[0]
    inp = packed[1:]
    pos = 0
    low, rng, code, scaled = 0, M32, 0, 0
    for _ in range(4):
        code = ((code << 8) | inp[pos]) & M32
        pos += 1
    length = width * height
    out = np.zeros(length, np.int64)
    res = np.zeros(length, np.int64)
    zc = np.ones(32, np.int64)
    o1 = np.ones(256 * 256, np.int64)
    o1t = np.full(256, 256, np.int64)
    o0 = np.full(256, 2, np.int64)
    o0t = 512
    used = np.empty(65536, np.int64)
    n_used = 0
    prev = 0
    for off in range(length):
        x = off % width
        y = off // width
        lft = 1 if x == 0 else int(res[off - 1] == 0)
        up = 1 if y == 0 else int(res[off - width] == 0)
        ul = 1 if (x == 0 or y == 0) else int(res[off - width - 1] == 0)
        ur = 1 if (y == 0 or x >= width - 1) else int(res[off - width + 1] == 0)
        zo = (lft | (up << 1) | (ul << 2) | (ur << 3)) << 1
        z, nz = zc[zo], zc[zo + 1]
        # decoder.get_freq(z + nz)
        tot = z + nz
        scaled = rng // tot
        val = ((code - low) & M32) // scaled
        if val >= tot:
            val = tot - 1
        bit = 0 if val < z else 1
        cum, freq = (0, z) if bit == 0 else (z, nz)
        low = (low + scaled * cum) & M32
        rng = scaled * freq
        while True:
            if ((low ^ ((low + rng) & M32)) & M32) < TOP:
                pass
            elif rng < BOTTOM:
                rng = (-low) & M32 & (BOTTOM - 1)
            else:
                break
            code = ((code << 8) | inp[pos]) & M32
            pos += 1
            low = (low << 8) & M32
            rng = (rng << 8) & M32
        zc[zo + bit] = (zc[zo + bit] + 32) & M32
        if zc[zo] + zc[zo + 1] >= 16384:
            zc[zo] = max(zc[zo] >> 1, 1)
            zc[zo + 1] = max(zc[zo + 1] >> 1, 1)
        sym = 0
        if bit == 1:
            above = res[off - width] if off >= width else 0
            key = (prev << 8) | above
            row = idx[key]
            if row < 0:
                row = n_used
                idx[key] = row
                used[n_used] = key
                n_used += 1
                pool[row, :] = 0
            fo = prev << 8
            excl = pool[row, 0] + o1[fo] + o0[0]
            tot = pool[row, 256] + o1t[prev] + o0t - excl
            scaled = rng // tot
            target = ((code - low) & M32) // scaled
            if target >= tot:
                target = tot - 1
            cum = 0
            sym = 1
            freq = pool[row, 1] + o1[fo + 1] + o0[1]
            while cum + freq <= target:
                cum += freq
                sym += 1
                freq = pool[row, sym] + o1[fo + sym] + o0[sym]
            low = (low + scaled * cum) & M32
            rng = scaled * freq
            while True:
                if ((low ^ ((low + rng) & M32)) & M32) < TOP:
                    pass
                elif rng < BOTTOM:
                    rng = (-low) & M32 & (BOTTOM - 1)
                else:
                    break
                code = ((code << 8) | inp[pos]) & M32
                pos += 1
                low = (low << 8) & M32
                rng = (rng << 8) & M32
            pool[row, sym] = (pool[row, sym] + 616) & 0xFFFF
            pool[row, 256] = (pool[row, 256] + 616) & 0xFFFF
            if pool[row, 256] >= 14 * 3584:
                t = 0
                for v in range(256):
                    pool[row, v] = pool[row, v] >> 1
                    t += pool[row, v]
                pool[row, 256] = t & 0xFFFF
            o1[fo + sym] = (o1[fo + sym] + 56) & 0xFFFF
            o1t[prev] = o1t[prev] + 56
            if o1t[prev] >= 8192:
                t = 0
                for v in range(256):
                    s2 = max(o1[fo + v] >> 1, 1)
                    o1[fo + v] = s2
                    t += s2
                o1t[prev] = t
            o0[sym] = (o0[sym] + 32) & 0xFFFF
            o0t += 32
            if o0t >= 4096:
                t = 0
                for v in range(256):
                    o0[v] = o0[v] >> 1
                    t += o0[v]
                o0t = t
        res[off] = sym
        # predictor
        left = out[off - 1] if x > 0 else 0
        upv = out[off - width] if y > 0 else 0
        if selector == 1:
            pred = upv
        elif selector == 2:
            pred = left
        elif selector == 3:
            pred = (left + upv) >> 1
        else:
            ulv = out[off - width - 1] if (x > 0 and y > 0) else 0
            if selector == 0:
                est = left + upv - ulv
                dl, du, dul = abs(est - left), abs(est - upv), abs(est - ulv)
                if dl <= du and dl <= dul:
                    pred = left
                elif du <= dul:
                    pred = upv
                else:
                    pred = ulv
            else:
                if ulv >= max(left, upv):
                    pred = min(left, upv)
                elif ulv <= min(left, upv):
                    pred = max(left, upv)
                else:
                    pred = left + upv - ulv
        delta = -((sym + 1) >> 1) if (sym & 1) else (sym >> 1)
        out[off] = (pred + delta) & 255
        prev = sym
    for i in range(n_used):
        idx[used[i]] = -1
    return out


@nb.njit(cache=True)
def _decode_one_bit(packed, width, height):
    """Port of meta_sam_parser._mask_codec._decode_raster (without the canonical re-encode check)."""
    pos = 0
    low, rng, code = 0, M32, 0
    for _ in range(4):
        code = ((code << 8) | packed[pos]) & M32
        pos += 1
    counts = np.ones(4096 * 2, np.int64)
    pw = width + 4
    padded = np.zeros(pw * (height + 2), np.int64)
    out = np.zeros(width * height, np.uint8)
    for y in range(height):
        rowp = (y + 2) * pw + 2
        for x in range(width):
            p = rowp + x
            pr, er = p - pw, p - 2 * pw
            ctx = (padded[p - 1] | (padded[p - 2] << 1) | (padded[pr - 2] << 2) | (padded[pr - 1] << 3)
                   | (padded[pr] << 4) | (padded[pr + 1] << 5) | (padded[pr + 2] << 6) | (padded[er - 2] << 7)
                   | (padded[er - 1] << 8) | (padded[er] << 9) | (padded[er + 1] << 10) | (padded[er + 2] << 11))
            o = ctx << 1
            z, one = counts[o], counts[o + 1]
            tot = z + one
            scaled = rng // tot
            val = ((code - low) & M32) // scaled
            if val >= tot:
                val = tot - 1
            bit = 0 if val < z else 1
            cum, freq = (0, z) if bit == 0 else (z, one)
            low = (low + scaled * cum) & M32
            rng = scaled * freq
            while True:
                if ((low ^ ((low + rng) & M32)) & M32) < TOP:
                    pass
                elif rng < BOTTOM:
                    rng = (-low) & M32 & (BOTTOM - 1)
                else:
                    break
                code = ((code << 8) | (packed[pos] if pos < len(packed) else 0)) & M32
                pos += 1
                low = (low << 8) & M32
                rng = (rng << 8) & M32
            padded[p] = bit
            out[y * width + x] = bit
            counts[o + bit] = (counts[o + bit] + 14) & 0xFFFF
            if counts[o] + counts[o + 1] >= 4096:
                counts[o] = max(counts[o] >> 1, 1)
                counts[o + 1] = max(counts[o + 1] >> 1, 1)
    return out


_IDX = np.full(65536, -1, np.int64)
_POOL = np.zeros((65536, 257), np.int64)  # 134 MB, allocated once per process


def decode(payload, width, height):
    """Binary mask (height x width, bool) from a ~lossless or !one_bit payload."""
    src = np.frombuffer(payload[1:].encode("ascii"), np.uint8)
    packed = _unpack(src, _VALUES)
    if payload[0] == "~":
        cov = _decode_lossless(packed, width, height, _IDX, _POOL)
        return (cov >= 129).reshape(height, width)
    return _decode_one_bit(packed, width, height).astype(bool).reshape(height, width)


class Obj:
    __slots__ = ("id", "box", "x0", "y0", "_m")

    def __init__(self, oid, box, x0, y0, m):
        self.id, self.box, self.x0, self.y0, self._m = oid, box, x0, y0, m

    def mask(self):
        """Bool mask in box space. Pixel (r, c) is image pixel (y0 + r, x0 + c)."""
        if isinstance(self._m, tuple):
            h, w, payload = self._m
            self._m = decode(payload, w, h)
        return self._m


def raw_text(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    parts = []
    with opener(path, "rt") as fh:
        for line in fh:
            e = json.loads(line)
            if e.get("type") == "response.output_text.delta":
                parts.append(e.get("delta", ""))
    return "".join(parts)


def read_frames(path):
    """Yield (frame_index, [Obj]) in frame order. Masks decode lazily."""
    text = raw_text(path)
    marks = list(_FRAME.finditer(text))
    for i, m in enumerate(marks):
        body = text[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        objs = []
        for o in _OBJ.finditer(body):
            x1, y1, x2, y2 = (float(v) for v in o.group(2, 3, 4, 5))
            box = (x1, y1, x2 + 1, y2 + 1)  # right and bottom exclusive, as in meta-sam-parser
            if o.group(6) is None:
                objs.append(Obj(o.group(1), box, int(box[0]), int(box[1]), None))
                continue
            x0 = int(box[0]) + int(o.group(6))
            y0 = int(box[1]) + int(o.group(7))
            objs.append(Obj(o.group(1), box, x0, y0, (int(o.group(8)), int(o.group(9)), o.group(10))))
        yield int(m.group(1)), objs


def _check(path):
    """Compare every box and mask with meta-sam-parser."""
    import sys
    import time
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sam_prompt_compare as spc
    spc.OUT = Path(path).parent
    prompt = Path(path).name[len("sam_"):-len("_events.jsonl")].replace("_", " ")
    boxes, masks = spc.load(prompt)
    t = time.perf_counter()
    n = bad = 0
    for f, objs in read_frames(path):
        for o in objs:
            ref = masks[f][o.id]
            full = np.zeros((720, 1280), bool)
            m = o.mask()
            full[o.y0:o.y0 + m.shape[0], o.x0:o.x0 + m.shape[1]] = m[:720 - o.y0, :1280 - o.x0]
            same = np.array_equal(full, spc.full_mask(ref) > 0) and np.allclose(o.box, boxes[f][o.id])
            n += 1
            bad += not same
    print(f"{path}: {n} masks, {bad} differ, {time.perf_counter() - t:.2f} s with sam_fast")


if __name__ == "__main__":
    import sys
    for p in sys.argv[2:] if sys.argv[1:2] == ["--check"] else []:
        _check(p)
