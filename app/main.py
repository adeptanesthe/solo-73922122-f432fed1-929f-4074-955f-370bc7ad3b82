"""FastAPI 应用：卷积转换 / 必读坐标 / 窗口求值 / 原模型参考 / 复核投票训练单。"""

from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException

from . import engine, jobs as jobs_mod, pack as pack_mod, schemas

app = FastAPI(
    title="边缘相机卷积转换 API",
    version="1.1.0",
    description=(
        "把逐通道 a*x+b 标准化折叠进 3x3 卷积，导出与图/窗口尺寸无关的参数包；"
        "按窗口列出必读原始像素坐标，并用原始像素直接求值（不生成标准化像素块）；"
        "支持质检员复核投票开单，按有效点计算标准化参数 a/b 的训练梯度。"
    ),
)
store = pack_mod.PackStore()
job_store = jobs_mod.JobStore()


def _http(exc: Exception) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "packs_cached": str(len(store)),
        "jobs_open": str(len(job_store)),
    }


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


# --------------------------------------------------------------------------- #
# 质检员复核投票训练单
# --------------------------------------------------------------------------- #
@app.post("/training-jobs", response_model=schemas.JobCreateResponse)
def create_training_job(req: schemas.JobCreateRequest) -> schemas.JobCreateResponse:
    """用现有卷积参数包和一张原始图（宽、高各 <= 64）开训练单。"""
    pack = _resolve(req)
    try:
        image = jobs_mod.validate_image(np.asarray(req.image, dtype=np.float64),
                                        int(pack["C"]))
        job = job_store.create(pack, image)
    except (jobs_mod.JobError, engine.EngineError) as exc:
        raise _http(exc) from exc
    return schemas.JobCreateResponse(
        job_id=job.job_id,
        H=job.H,
        W=job.W,
        C=job.C,
        O=job.O,
        message="训练单已创建，等待质检员提交复核票",
    )


@app.post("/training-jobs/ticket", response_model=schemas.TicketResponse)
def submit_ticket(req: schemas.TicketRequest) -> schemas.TicketResponse:
    """整份提交/替换某质检员的票；空 ticket 即撤回。失败保留原票与计算结果。"""
    try:
        job = job_store.get(req.job_id)
    except jobs_mod.JobError as exc:
        raise _http(exc) from exc
    try:
        normalized = jobs_mod.normalize_ticket(
            [item.model_dump() for item in req.ticket],
            job.H, job.W, job.O,
        )
        receipt = job.replace_ticket(req.inspector, normalized)
    except (jobs_mod.JobError, engine.EngineError) as exc:
        # replace_ticket 内部已做快照回滚；此处原票与累加量保持提交前状态
        raise _http(exc) from exc
    return schemas.TicketResponse(**receipt)


@app.post("/training-jobs/query", response_model=schemas.JobQueryResponse)
def query_training_job(req: schemas.JobQueryRequest) -> schemas.JobQueryResponse:
    """查单：有效点数、MSE 与各输入通道 a/b 的梯度；无有效点提示补标。"""
    try:
        job = job_store.get(req.job_id)
    except jobs_mod.JobError as exc:
        raise _http(exc) from exc
    return schemas.JobQueryResponse(**job.query())


@app.get("/training-jobs/{job_id}", response_model=schemas.JobDetailResponse)
def get_training_job(job_id: str) -> schemas.JobDetailResponse:
    """查看训练单上各质检员当前已提交的票（只读）。"""
    try:
        job = job_store.get(job_id)
    except jobs_mod.JobError as exc:
        raise _http(exc) from exc
    all_tickets = job.snapshot_tickets()
    return schemas.JobDetailResponse(
        job_id=job.job_id,
        H=job.H,
        W=job.W,
        C=job.C,
        O=job.O,
        n_eff=job.query()["n_eff"],
        inspectors=job.inspectors(),
        tickets=[
            schemas.JobTicket(
                inspector=inspector,
                ticket=[
                    {"r": p, "c": q, "o": o, "label": label}
                    for (p, q, o), label in sorted(ticket.items())
                ],
                ticket_size=len(ticket),
            )
            for inspector, ticket in sorted(all_tickets.items())
        ],
    )
