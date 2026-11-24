"""参数包：把逐通道仿射标准化折叠进卷积权重。

原模型（H x W x C 原图 X，float64 运算）：

    Y[h, w, c] = a[c] * X[h, w, c] + b[c]                # 图内标准化
    Y_pad      = Y 外圈补一圈零
    Z[p,q,o]   = bias[o] + sum_{c,u,v} K[o,c,u,v] * Y_pad[p+u, q+v, c]

展开标准化并按“源像素是否在图内”拆分：

    Z[p,q,o] = bias[o]
             + sum_{图内 tap} A[o,c,u,v] * X[源r, 源c, c]     # A = K*a
             + sum_{图内 tap} t[o,c,u,v]                      # t = K*b

参数包只含 C/O/3x3 尺寸的数组，与整图尺寸、窗口尺寸完全无关，
换图换窗口直接复用。求值端不需要、也不会生成标准化像素块。
"""

from __future__ import annotations

import base64
import io
from collections import OrderedDict
from typing import Dict, Mapping, Tuple

import numpy as np

PACK_VERSION = 1
MAX_CHANNELS = 4
MAX_ABS = 16.0
_STORE_LIMIT = 128


class PackError(ValueError):
    """参数包构建/解析错误。"""


def _check_abs(name: str, arr: np.ndarray, limit: float = MAX_ABS) -> None:
    if np.any(np.abs(arr) > limit):
        raise PackError(f"{name} 中存在绝对值超过 {limit:g} 的元素")


def build_pack(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    kernel: np.ndarray | list,
    bias: np.ndarray | list[float],
) -> Dict[str, np.ndarray]:
    """校验原始模型参数并折叠成参数包。

    Returns
    -------
    dict，键均为 np.ndarray（float64/int64）：
        version, C, O, a, b, K, bias      原始参数（参考实现/复核用）
        A = K*a, t = K*b                  折叠后的部署参数
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    k_arr = np.asarray(kernel, dtype=np.float64)
    bias_arr = np.asarray(bias, dtype=np.float64)

    if a_arr.ndim != 1 or b_arr.shape != a_arr.shape:
        raise PackError("a、b 必须是等长的一维逐通道参数")
    c = a_arr.shape[0]
    if not 1 <= c <= MAX_CHANNELS:
        raise PackError(f"输入通道数 C 必须在 1..{MAX_CHANNELS} 之间，收到 {c}")
    if k_arr.ndim != 4 or k_arr.shape[1:] != (c, 3, 3):
        raise PackError(f"kernel 形状必须为 (O, {c}, 3, 3)，收到 {k_arr.shape}")
    o = k_arr.shape[0]
    if not 1 <= o <= MAX_CHANNELS:
        raise PackError(f"输出通道数 O 必须在 1..{MAX_CHANNELS} 之间，收到 {o}")
    if bias_arr.shape != (o,):
        raise PackError(f"bias 长度必须为 {o}，收到 {bias_arr.shape}")

    for name, arr in (("a", a_arr), ("b", b_arr), ("kernel", k_arr), ("bias", bias_arr)):
        _check_abs(name, arr)

    # 折叠：A[o,c,u,v] = K*a[c]，t[o,c,u,v] = K*b[c]
    a_pack = k_arr * a_arr[None, :, None, None]
    t_pack = k_arr * b_arr[None, :, None, None]

    return {
        "version": np.int64(PACK_VERSION),
        "C": np.int64(c),
        "O": np.int64(o),
        "a": a_arr,
        "b": b_arr,
        "K": k_arr,
        "bias": bias_arr,
        "A": a_pack,
        "t": t_pack,
    }


def encode_pack(pack: Mapping[str, np.ndarray]) -> str:
    """参数包 -> base64 字符串（np.savez 二进制，自包含、可离线复用）。"""
    buf = io.BytesIO()
    np.savez(buf, **dict(pack))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_pack(token: str) -> Dict[str, np.ndarray]:
    """base64 字符串 -> 参数包，并做版本校验。"""
    try:
        raw = base64.b64decode(token.encode("ascii"), validate=True)
        with np.load(io.BytesIO(raw)) as z:
            pack = {k: z[k] for k in z.files}
    except Exception as exc:  # noqa: BLE001 - 统一转成 PackError
        raise PackError(f"参数包解析失败: {exc}") from exc
    if int(pack.get("version", np.int64(0))) != PACK_VERSION:
        raise PackError("参数包版本不兼容")
    return pack


class PackStore:
    """进程内 LRU 参数包存储（按 id 复用，避免重复传大包）。"""

    def __init__(self, limit: int = _STORE_LIMIT) -> None:
        self._items: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        self._limit = limit

    def put(self, pack_id: str, pack: Dict[str, np.ndarray]) -> None:
        self._items[pack_id] = pack
        self._items.move_to_end(pack_id)
        while len(self._items) > self._limit:
            self._items.popitem(last=False)

    def get(self, pack_id: str) -> Dict[str, np.ndarray]:
        try:
            self._items.move_to_end(pack_id)
        except KeyError:
            raise PackError(f"未知 pack_id: {pack_id}") from None
        return self._items[pack_id]

    def __len__(self) -> int:
        return len(self._items)


def resolve_pack(
    store: PackStore,
    pack_token: str | None,
    pack_id: str | None,
) -> Tuple[Dict[str, np.ndarray], str]:
    """二选一解析参数包：优先内联 pack，其次 pack_id。返回 (pack, 来源)。"""
    if pack_token is not None and pack_id is not None:
        raise PackError("pack 与 pack_id 只能二选一")
    if pack_token is not None:
        return decode_pack(pack_token), "pack"
    if pack_id is not None:
        return store.get(pack_id), "pack_id"
    raise PackError("必须提供 pack 或 pack_id 之一")
