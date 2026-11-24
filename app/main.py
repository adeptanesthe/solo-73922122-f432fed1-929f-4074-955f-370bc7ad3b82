"""FastAPI 应用：卷积转换 / 必读坐标 / 窗口求值 / 原模型参考。"""

from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException

from . import engine, pack as pack_mod, schemas

app = FastAPI(
    title="边缘相机卷积转换 API",
    version="1.0.0",
    description=(
        "把逐通道 a*x+b 标准化折叠进 3x3 卷积，导出与图/窗口尺寸无关的参数包；"
        "按窗口列出必读原始像素坐标，并用原始像素直接求值（不生成标准化像素块）。"
    ),
)
store = pack_mod.PackStore()


def _http(exc: Exception) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "packs_cached": str(len(store))}


@app.post("/convert", response_model=schemas.ConvertResponse)
def convert(req: schemas.ConvertRequest) -> schemas.ConvertResponse:
    """原始模型参数 -> 折叠参数包（A=K*a, t=K*b），base64 导出并缓存 id。"""
    try:
        pack = pack_mod.build_pack(req.a, req.b, req.kernel, req.bias)
        token = pack_mod.encode_pack(pack)
        pack_id = "pack_" + hashlib.sha256(token.encode("ascii")).hexdigest()[:16]
        store.put(pack_id, pack)
    except pack_mod.PackError as exc:
        raise _http(exc) from exc
    return schemas.ConvertResponse(
        pack=token,
        pack_id=pack_id,
        C=int(pack["C"]),
        O=int(pack["O"]),
        folded_A_shape=list(pack["A"].shape),
        folded_t_shape=list(pack["t"].shape),
    )


def _resolve(req: schemas._PackRef) -> Dict[str, np.ndarray]:
    try:
        pack, _src = pack_mod.resolve_pack(store, req.pack, req.pack_id)
    except pack_mod.PackError as exc:
        raise _http(exc) from exc
    return pack


@app.post("/required-coordinates", response_model=schemas.CoordsResponse)
def required_coordinates(req: schemas.CoordsRequest) -> schemas.CoordsResponse:
    """列出窗口输出在数学上依赖的全部图内必读坐标。"""
    pack = _resolve(req)
    try:
        result = engine.required_coordinates(
            pack, req.H, req.W, *req.window.as_tuple()
        )
    except engine.EngineError as exc:
        raise _http(exc) from exc
    return schemas.CoordsResponse(
        coords=[list(rc) for rc in result["coords"]],
        by_channel={
            k: {"index": v["index"], "coords": [list(rc) for rc in v["coords"]]}
            for k, v in result["by_channel"].items()
        },
        active_channels=result["active_channels"],
        count=result["count"],
        note=result["note"],
    )


@app.post("/evaluate", response_model=schemas.EvalResponse)
def evaluate(req: schemas.EvalRequest) -> schemas.EvalResponse:
    """仅用必读原始像素计算窗口响应；缺任一必读坐标返回 422。"""
    pack = _resolve(req)
    pixels = {}
    for key, val in req.pixels.items():
        r, c = (int(x) for x in key.split(","))
        pixels[(r, c)] = val
    try:
        out = engine.evaluate_window(
            pack, req.H, req.W, *req.window.as_tuple(), pixels
        )
    except engine.EngineError as exc:
        raise _http(exc) from exc
    return schemas.EvalResponse(
        window=list(req.window.as_tuple()),
        shape=list(out.shape),
        output=out.tolist(),
        pixels_used=len(pixels),
    )


@app.post("/reference", response_model=schemas.ReferenceResponse)
def reference(req: schemas.ReferenceRequest) -> schemas.ReferenceResponse:
    """原模型整图参考实现（逐像素标准化+补零+互相关），仅用于误差核对。"""
    pack = _resolve(req)
    try:
        out = engine.reference_full_image(pack, np.asarray(req.image, dtype=np.float64))
    except engine.EngineError as exc:
        raise _http(exc) from exc
    return schemas.ReferenceResponse(shape=list(out.shape), output=out.tolist())
