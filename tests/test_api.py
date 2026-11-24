"""FastAPI 端到端测试（TestClient），含边角示例流程与错误路径。"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import engine
from app.main import app

client = TestClient(app)


def corner_model():
    # 2 输入通道 / 3 输出通道；通道 1 的 a=0（标准化退化为常量）
    a = [1.5, 0.0]
    b = [-2.0, 0.75]
    K = np.zeros((3, 2, 3, 3))
    K[0, 0, 1, 1] = 1.0
    K[1, :, :, :] = np.array(
        [[[0.25, 0, -0.5], [0, 1.0, 0], [0.5, 0, -0.25]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    K[2, 1, 0, 0] = -1.25  # 仅作用在 a=0 通道：该通道仍因 b 影响输出，但无需读像素
    bias = [0.5, -1.0, 2.0]
    return a, b, K.tolist(), bias


@pytest.fixture(scope="module")
def converted():
    r = client.post("/convert", json=dict(zip(
        ("a", "b", "kernel", "bias"), corner_model())))
    assert r.status_code == 200, r.text
    return r.json()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_convert_validation_errors():
    a, b, K, bias = corner_model()
    bad_a = a + [1.0]  # 通道数不一致
    r = client.post("/convert", json={"a": bad_a, "b": b, "kernel": K, "bias": bias})
    assert r.status_code == 422
    r = client.post("/convert", json={
        "a": [17.0], "b": [0.0],
        "kernel": np.zeros((1, 1, 3, 3)).tolist(), "bias": [0.0]})
    assert r.status_code == 422


def test_end_to_end_corner_window_matches_reference(converted):
    H, W = 4, 3
    rng = np.random.default_rng(42)
    img = rng.uniform(-16, 16, (H, W, 2))
    win = (0, 0, 2, 2)  # 左上角边角窗口

    r = client.post("/required-coordinates", json={
        "pack_id": converted["pack_id"], "H": H, "W": W,
        "window": dict(zip(("r0", "c0", "r1", "c1"), win))})
    assert r.status_code == 200, r.text
    coords = r.json()
    assert r.json()["active_channels"] == [0]  # 通道 1 的 a=0，不读像素

    pixels = {f"{rr},{cc}": img[rr, cc].tolist() for rr, cc in coords["coords"]}
    r = client.post("/evaluate", json={
        "pack": converted["pack"], "H": H, "W": W,
        "window": dict(zip(("r0", "c0", "r1", "c1"), win)),
        "pixels": pixels})
    assert r.status_code == 200, r.text
    out = np.array(r.json()["output"])

    r = client.post("/reference", json={"pack": converted["pack"],
                                        "image": img.tolist()})
    ref = np.array(r.json()["output"])
    np.testing.assert_allclose(out, ref[0:2, 0:2], atol=1e-10, rtol=0)
    assert np.abs(out - ref[0:2, 0:2]).max() <= 1e-8


def test_missing_pixel_returns_422(converted):
    H = W = 3
    win = (1, 1, 2, 2)
    r = client.post("/required-coordinates", json={
        "pack_id": converted["pack_id"], "H": H, "W": W,
        "window": dict(zip(("r0", "c0", "r1", "c1"), win))})
    coords = r.json()["coords"]
    pixels = {f"{rr},{cc}": [0.1, 0.2] for rr, cc in coords}
    del pixels[f"{coords[0][0]},{coords[0][1]}"]
    r = client.post("/evaluate", json={
        "pack_id": converted["pack_id"], "H": H, "W": W,
        "window": dict(zip(("r0", "c0", "r1", "c1"), win)),
        "pixels": pixels})
    assert r.status_code == 422 and "缺少必读像素" in r.json()["detail"]


def test_unknown_pack_id_422():
    r = client.post("/required-coordinates", json={
        "pack_id": "pack_nope", "H": 3, "W": 3,
        "window": {"r0": 0, "c0": 0, "r1": 1, "c1": 1}})
    assert r.status_code == 422


def test_single_row_single_column_image(converted):
    H, W = 1, 5
    img = np.linspace(-16, 16, H * W * 2).reshape(H, W, 2)
    win = (0, 0, 1, W)
    r = client.post("/required-coordinates", json={
        "pack": converted["pack"], "H": H, "W": W,
        "window": dict(zip(("r0", "c0", "r1", "c1"), win))})
    coords = r.json()["coords"]
    pixels = {f"{rr},{cc}": img[rr, cc].tolist() for rr, cc in coords}
    r = client.post("/evaluate", json={
        "pack": converted["pack"], "H": H, "W": W,
        "window": dict(zip(("r0", "c0", "r1", "c1"), win)),
        "pixels": pixels})
    out = np.array(r.json()["output"])
    ref = np.array(client.post("/reference", json={
        "pack": converted["pack"], "image": img.tolist()}).json()["output"])
    np.testing.assert_allclose(out, ref, atol=1e-10, rtol=0)


def test_pack_size_independent_of_image_size():
    a, b, K, bias = corner_model()
    r1 = client.post("/convert", json={"a": a, "b": b, "kernel": K, "bias": bias})
    token = r1.json()["pack"]
    # 重复转换得到完全相同的参数包字节
    r2 = client.post("/convert", json={"a": a, "b": b, "kernel": K, "bias": bias})
    assert r2.json()["pack"] == token
    # 参数包体积不随图像尺寸变化
    import base64
    raw_len = len(base64.b64decode(token))
    assert raw_len < 4096


# --------------------------------------------------------------------------- #
# 质检员复核投票训练单（端到端）
# --------------------------------------------------------------------------- #
TRAIN_H, TRAIN_W = 8, 6


def _training_pack():
    # 通道 1 的 a=0：该通道不读像素，但训练梯度仍须对其 a、b 求导
    return {
        "a": [1.2, 0.0],
        "b": [-0.5, 0.8],
        "kernel": np.array(
            [[[[0.5, -0.25, 0.0], [0.0, 1.0, 0.25], [0.5, 0.0, -0.5]],
              [[0.0, 0.25, 0.0], [-0.5, 0.0, 0.75], [0.0, 0.0, 0.5]]],
             [[[0.0, 0.0, 1.0], [-1.0, 0.5, 0.0], [0.0, 0.25, 0.0]],
              [[0.75, 0.0, -0.25], [0.0, 1.0, 0.0], [-0.5, 0.25, 0.0]]]]
        ).tolist(),
        "bias": [0.3, -0.7],
    }


@pytest.fixture(scope="module")
def job_created():
    conv = client.post("/convert", json=_training_pack())
    assert conv.status_code == 200, conv.text
    pack_id = conv.json()["pack_id"]
    rng = np.random.default_rng(7)
    image = rng.uniform(-8, 8, (TRAIN_H, TRAIN_W, 2)).tolist()
    r = client.post("/training-jobs",
                    json={"pack_id": pack_id, "image": image})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["H"] == TRAIN_H and body["W"] == TRAIN_W
    assert body["C"] == 2 and body["O"] == 2
    return body["job_id"], image


def _vote(r, c, o, label):
    return {"r": r, "c": c, "o": o, "label": label}


def test_create_job_rejects_oversized_image():
    conv = client.post("/convert", json=_training_pack())
    pack = conv.json()["pack"]
    r = client.post("/training-jobs",
                    json={"pack": pack, "image": np.zeros((65, 8, 2)).tolist()})
    assert r.status_code == 422
    r = client.post("/training-jobs",
                    json={"pack": pack, "image": np.zeros((8, 65, 2)).tolist()})
    assert r.status_code == 422


def test_query_empty_job_prompts_relabel(job_created):
    job_id, _ = job_created
    r = client.post("/training-jobs/query", json={"job_id": job_id})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["has_effective_points"] is False and body["n_eff"] == 0
    assert body["mse"] is None and body["grad_a"] is None and body["grad_b"] is None
    assert "补标" in body["message"]


def test_full_ticket_lifecycle_and_withdraw_changes_gradient(job_created):
    job_id, image = job_created
    img = np.asarray(image, dtype=np.float64)

    # alice 投两张缺陷票、一张良品票
    r = client.post("/training-jobs/ticket", json={
        "job_id": job_id, "inspector": "alice", "ticket": [
            _vote(0, 0, 0, "defect"), _vote(3, 2, 1, "defect"),
            _vote(7, 5, 0, "good")]})
    assert r.status_code == 200, r.text
    assert r.json()["action"] == "submitted" and r.json()["ticket_size"] == 3
    assert r.json()["points_touched"] == 3

    q1 = client.post("/training-jobs/query", json={"job_id": job_id}).json()
    assert q1["has_effective_points"] and q1["n_eff"] == 3
    assert len(q1["grad_a"]) == 2 and len(q1["grad_b"]) == 2
    # a=0 的通道 1 仍必须给出梯度
    assert q1["grad_a"][1] is not None

    # 与引擎直算结果逐值核对（目标：缺陷 +1、良品 -1）
    from app import engine, pack as pack_mod
    pack = pack_mod.build_pack(**_training_pack())
    targets = {(0, 0, 0): 1, (3, 2, 1): 1, (7, 5, 0): -1}
    stats = engine.training_stats_from_targets(pack, img, targets)
    assert q1["mse"] == pytest.approx(stats["mse"], rel=1e-11, abs=1e-10)
    np.testing.assert_allclose(q1["grad_a"], stats["grad_a"], rtol=1e-11, atol=1e-9)
    np.testing.assert_allclose(q1["grad_b"], stats["grad_b"], rtol=1e-11, atol=1e-9)

    # bob 对其中两点投反对票：(0,0,0) 平票失效，(3,2,1) 平票失效，剩 1 个有效点
    r = client.post("/training-jobs/ticket", json={
        "job_id": job_id, "inspector": "bob", "ticket": [
            _vote(0, 0, 0, "good"), _vote(3, 2, 1, "good")]})
    assert r.status_code == 200 and r.json()["points_touched"] == 2
    q2 = client.post("/training-jobs/query", json={"job_id": job_id}).json()
    assert q2["n_eff"] == 1
    assert q2["mse"] != q1["mse"]  # 有效点集合变了，统计随之变化

    # alice 撤票：仅剩 bob 的两张良品票，全部有效且目标为 -1
    r = client.post("/training-jobs/ticket", json={
        "job_id": job_id, "inspector": "alice", "ticket": []})
    assert r.status_code == 200 and r.json()["action"] == "withdrawn"
    q3 = client.post("/training-jobs/query", json={"job_id": job_id}).json()
    assert q3["n_eff"] == 2
    stats3 = engine.training_stats_from_targets(
        pack, img, {(0, 0, 0): -1, (3, 2, 1): -1})
    assert q3["mse"] == pytest.approx(stats3["mse"], rel=1e-11, abs=1e-10)
    np.testing.assert_allclose(q3["grad_a"], stats3["grad_a"], rtol=1e-11, atol=1e-9)

    # bob 也撤票：无有效点，恢复补标提示
    client.post("/training-jobs/ticket", json={
        "job_id": job_id, "inspector": "bob", "ticket": []})
    q4 = client.post("/training-jobs/query", json={"job_id": job_id}).json()
    assert q4["has_effective_points"] is False and "补标" in q4["message"]

    detail = client.get(f"/training-jobs/{job_id}").json()
    assert detail["inspectors"] == [] and detail["tickets"] == []


def test_ticket_replacement_is_whole_document(job_created):
    job_id, image = job_created
    client.post("/training-jobs/ticket", json={
        "job_id": job_id, "inspector": "carol", "ticket": [
            _vote(1, 1, 0, "defect"), _vote(2, 2, 1, "good")]})
    # 整份替换：旧票中未出现的点被删除
    r = client.post("/training-jobs/ticket", json={
        "job_id": job_id, "inspector": "carol", "ticket": [
            _vote(1, 1, 0, "good")]})
    assert r.status_code == 200 and r.json()["action"] == "replaced"
    detail = client.get(f"/training-jobs/{job_id}").json()
    carol = next(t for t in detail["tickets"] if t["inspector"] == "carol")
    assert carol["ticket_size"] == 1
    assert carol["ticket"][0] == _vote(1, 1, 0, "good")
    client.post("/training-jobs/ticket", json={
        "job_id": job_id, "inspector": "carol", "ticket": []})


def test_invalid_votes_rejected_without_state_change(job_created):
    job_id, _ = job_created
    before = client.get(f"/training-jobs/{job_id}").json()
    for bad in [
        [_vote(TRAIN_H, 0, 0, "defect")],       # 行越界
        [_vote(0, TRAIN_W, 0, "defect")],       # 列越界
        [_vote(0, 0, 2, "defect")],             # 输出通道越界
        [_vote(0, 0, 0, "maybe")],              # 非法标签
        [_vote(0, 0, 0, "defect"),
         _vote(0, 0, 0, "good")],               # 同人同点两票
    ]:
        r = client.post("/training-jobs/ticket", json={
            "job_id": job_id, "inspector": "dave", "ticket": bad})
        assert r.status_code == 422, r.text
    after = client.get(f"/training-jobs/{job_id}").json()
    assert after["tickets"] == before["tickets"]  # 失败不留痕


def test_unknown_job_id_422():
    assert client.post("/training-jobs/query",
                       json={"job_id": "job_nope"}).status_code == 422
    assert client.post("/training-jobs/ticket", json={
        "job_id": "job_nope", "inspector": "x",
        "ticket": []}).status_code == 422
    assert client.get("/training-jobs/job_nope").status_code == 422


def test_tie_points_excluded_and_majority_flips_target(job_created):
    job_id, image = job_created
    key_vote = _vote(4, 4, 0, "defect")
    client.post("/training-jobs/ticket", json={
        "job_id": job_id, "inspector": "e1", "ticket": [key_vote]})
    q = client.post("/training-jobs/query", json={"job_id": job_id}).json()
    assert q["n_eff"] == 1  # 缺陷 1:0 -> 目标 +1

    client.post("/training-jobs/ticket", json={
        "job_id": job_id, "inspector": "e2",
        "ticket": [_vote(4, 4, 0, "good")]})
    q = client.post("/training-jobs/query", json={"job_id": job_id}).json()
    assert q["n_eff"] == 0  # 1:1 平票剔除

    client.post("/training-jobs/ticket", json={
        "job_id": job_id, "inspector": "e3",
        "ticket": [_vote(4, 4, 0, "good")]})
    q = client.post("/training-jobs/query", json={"job_id": job_id}).json()
    assert q["n_eff"] == 1  # 良品 2:1 -> 目标 -1，重新有效

    for name in ("e1", "e2", "e3"):
        client.post("/training-jobs/ticket", json={
            "job_id": job_id, "inspector": name, "ticket": []})
