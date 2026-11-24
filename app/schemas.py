"""Pydantic v2 请求/响应模型与输入约束。

所有数值约束：输入/输出通道 1..4；像素与模型参数绝对值 <= 16；a 允许为 0。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

MAX_CHANNELS = 4
MAX_ABS = 16.0


def _bounded_3d(name: str, v, limit: float = MAX_ABS):
    arr = np.asarray(v, dtype=np.float64)
    if np.any(np.abs(arr) > limit):
        raise ValueError(f"{name} 存在绝对值超过 {limit:g} 的元素")
    return v


class ConvertRequest(BaseModel):
    """原始模型参数：逐通道 a*x+b、3x3 卷积权重 K 和偏置 bias。"""

    a: List[float] = Field(..., description="逐通道缩放，长度 C，允许为 0")
    b: List[float] = Field(..., description="逐通道偏移，长度 C")
    kernel: List[List[List[List[float]]]] = Field(
        ..., description="卷积权重，形状 (O, C, 3, 3)"
    )
    bias: List[float] = Field(..., description="输出偏置，长度 O")

    @field_validator("a", "b")
    @classmethod
    def _check_ab(cls, v):
        return _bounded_3d("a/b", v)

    @field_validator("kernel")
    @classmethod
    def _check_kernel(cls, v):
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 4 or arr.shape[2:] != (3, 3):
            raise ValueError("kernel 形状必须为 (O, C, 3, 3)")
        if not 1 <= arr.shape[0] <= MAX_CHANNELS or not 1 <= arr.shape[1] <= MAX_CHANNELS:
            raise ValueError(f"输入/输出通道数必须在 1..{MAX_CHANNELS} 之间")
        if np.any(np.abs(arr) > MAX_ABS):
            raise ValueError(f"kernel 存在绝对值超过 {MAX_ABS:g} 的元素")
        return v

    @field_validator("bias")
    @classmethod
    def _check_bias(cls, v):
        return _bounded_3d("bias", v)

    @model_validator(mode="after")
    def _check_shapes(self):
        c = len(self.a)
        if len(self.b) != c:
            raise ValueError("a 与 b 长度必须一致")
        k = np.asarray(self.kernel, dtype=np.float64)
        if k.shape[1] != c:
            raise ValueError(f"kernel 第 2 维必须等于输入通道数 {c}")
        if len(self.bias) != k.shape[0]:
            raise ValueError("bias 长度必须等于输出通道数 O")
        return self


class ConvertResponse(BaseModel):
    pack: str = Field(..., description="base64 参数包（自包含，可离线复用）")
    pack_id: str
    C: int
    O: int
    folded_A_shape: List[int]
    folded_t_shape: List[int]


class Window(BaseModel):
    r0: int = Field(..., ge=0)
    c0: int = Field(..., ge=0)
    r1: int = Field(..., gt=0)
    c1: int = Field(..., gt=0)

    def as_tuple(self):
        return self.r0, self.c0, self.r1, self.c1


class _PackRef(BaseModel):
    pack: Optional[str] = Field(None, description="内联 base64 参数包")
    pack_id: Optional[str] = Field(None, description="此前 /convert 返回的 id")

    @model_validator(mode="after")
    def _exactly_one(self):
        if (self.pack is None) == (self.pack_id is None):
            raise ValueError("pack 与 pack_id 必须且只能提供一个")
        return self


class CoordsRequest(_PackRef):
    H: int = Field(..., gt=0)
    W: int = Field(..., gt=0)
    window: Window


class CoordsResponse(BaseModel):
    coords: List[List[int]]
    by_channel: Dict[str, dict]
    active_channels: List[int]
    count: int
    note: str


class EvalRequest(_PackRef):
    H: int = Field(..., gt=0)
    W: int = Field(..., gt=0)
    window: Window
    pixels: Dict[str, List[float]] = Field(
        ...,
        description='原始像素，键 "r,c"，值为长度 C 的逐通道原始值（未标准化）',
    )

    @field_validator("pixels")
    @classmethod
    def _check_pixels(cls, v):
        seen = set()
        for key, val in v.items():
            parts = key.split(",")
            if len(parts) != 2:
                raise ValueError(f'像素键必须形如 "r,c"，收到 {key!r}')
            try:
                rc = (int(parts[0]), int(parts[1]))
            except ValueError:
                raise ValueError(f"像素键坐标必须为整数: {key!r}") from None
            if rc in seen:
                raise ValueError(f"重复像素坐标: {key!r}")
            seen.add(rc)
            if len(val) == 0 or len(val) > MAX_CHANNELS:
                raise ValueError(f"像素 {key!r} 通道数必须在 1..{MAX_CHANNELS} 之间")
            if any(abs(x) > MAX_ABS for x in val):
                raise ValueError(f"像素 {key!r} 存在绝对值超过 {MAX_ABS:g} 的值")
        if not v:
            raise ValueError("pixels 不能为空")
        return v


class EvalResponse(BaseModel):
    window: List[int]
    shape: List[int]
    output: List[List[List[float]]]
    pixels_used: int


class ReferenceRequest(_PackRef):
    """原模型整图参考计算（仅误差核对用，会显式生成标准化块）。"""

    image: List[List[List[float]]]

    @field_validator("image")
    @classmethod
    def _check_image(cls, v):
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError("image 形状必须为 (H, W, C)")
        if not 1 <= arr.shape[2] <= MAX_CHANNELS:
            raise ValueError(f"通道数必须在 1..{MAX_CHANNELS} 之间")
        if np.any(np.abs(arr) > MAX_ABS):
            raise ValueError(f"像素存在绝对值超过 {MAX_ABS:g} 的值")
        return v


class ReferenceResponse(BaseModel):
    shape: List[int]
    output: List[List[List[float]]]


# --------------------------------------------------------------------------- #
# 质检员复核投票训练单
# --------------------------------------------------------------------------- #
MAX_SIDE = 64


class JobCreateRequest(_PackRef):
    """开单：现有卷积参数包 + 一张原始图（宽、高各至多 64 像素）。"""

    image: List[List[List[float]]] = Field(
        ..., description="原始图 (H, W, C)，H、W 均在 1..64 之间，像素绝对值 <= 16"
    )

    @field_validator("image")
    @classmethod
    def _check_image(cls, v):
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError("image 形状必须为 (H, W, C)")
        H, W, C = arr.shape
        if not 1 <= H <= MAX_SIDE or not 1 <= W <= MAX_SIDE:
            raise ValueError(f"原图高、宽必须在 1..{MAX_SIDE} 之间，收到 {H}x{W}")
        if not 1 <= C <= MAX_CHANNELS:
            raise ValueError(f"通道数必须在 1..{MAX_CHANNELS} 之间")
        if np.any(np.abs(arr) > MAX_ABS):
            raise ValueError(f"像素存在绝对值超过 {MAX_ABS:g} 的值")
        return v


class JobCreateResponse(BaseModel):
    job_id: str
    H: int
    W: int
    C: int
    O: int
    message: str


class VoteItem(BaseModel):
    """单条票：输出点 (r, c, o) 与标签（defect=缺陷 / good=良品）。"""

    r: int = Field(..., ge=0, description="输出点行")
    c: int = Field(..., ge=0, description="输出点列")
    o: int = Field(..., ge=0, description="输出通道")
    label: str = Field(..., description="defect=缺陷票，good=良品票")


class TicketRequest(BaseModel):
    """整份提交/替换某质检员的票；ticket 为空数组即撤回。"""

    job_id: str
    inspector: str = Field(..., min_length=1, max_length=64)
    ticket: List[VoteItem] = Field(
        ..., description="该质检员本次的整份票；空数组表示撤回自己的全部票"
    )


class TicketResponse(BaseModel):
    job_id: str
    inspector: str
    action: str = Field(..., description="submitted / replaced / withdrawn")
    ticket_size: int
    points_touched: int
    n_eff: int


class JobQueryRequest(BaseModel):
    job_id: str


class JobQueryResponse(BaseModel):
    job_id: str
    has_effective_points: bool
    n_eff: int
    mse: Optional[float] = None
    grad_a: Optional[List[float]] = Field(
        None, description="各输入通道 a 的梯度 ∂L/∂a；无有效点时为 null"
    )
    grad_b: Optional[List[float]] = Field(
        None, description="各输入通道 b 的梯度 ∂L/∂b；无有效点时为 null"
    )
    message: str


class JobTicket(BaseModel):
    inspector: str
    ticket: List[VoteItem]
    ticket_size: int


class JobDetailResponse(BaseModel):
    job_id: str
    H: int
    W: int
    C: int
    O: int
    n_eff: int
    inspectors: List[str]
    tickets: List[JobTicket]
