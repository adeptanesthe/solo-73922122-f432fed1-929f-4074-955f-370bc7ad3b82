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
