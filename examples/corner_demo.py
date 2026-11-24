"""可调用的边角示例：走完整 API 流程并与原模型整图计算核对。

用法
----
# 1) 启动服务（Docker Compose 或本地）：
#    docker compose up --build        # 或  uvicorn app.main:app --host 0.0.0.0 --port 8000
# 2) 运行示例：
python examples/corner_demo.py                 # 默认 http://127.0.0.1:8000
python examples/corner_demo.py --offline       # 不起服务，直接调用引擎
python examples/corner_demo.py --base-url http://其他地址:8000

场景
----
4x3 小图、2 输入通道（通道 1 的 a=0，标准化退化为常量）、3 输出通道；
分别请求左上角、右下角、整图三个窗口（边角补零最密集），
只用 /required-coordinates 返回的原始像素调用 /evaluate，
再与 /reference 的原模型整图结果逐值比对，要求最大绝对误差 <= 1e-8。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import engine, pack as pack_mod  # noqa: E402

# ---------------------------------------------------------------- 模型与图
A_SCALE = [1.5, 0.0]          # 逐通道 a*x+b；通道 1 的 a=0
B_SHIFT = [-2.0, 0.75]
KERNEL = [
    # 输出通道 0：仅中心 tap
    [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
     [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
    # 输出通道 1：多 tap 混合
    [[[0.25, 0.0, -0.5], [0.0, 1.0, 0.0], [0.5, 0.0, -0.25]],
     [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
    # 输出通道 2：只挂在 a=0 的通道 1 上 -> 不读像素，仅常量贡献
    [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
     [[-1.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
]
BIAS = [0.5, -1.0, 2.0]
H, W = 4, 3
# 固定像素值，保证示例可复现（绝对值均 <= 16）
IMAGE = [
    [[1.0, 2.0], [-3.5, 4.0], [7.0, -8.0]],
    [[0.5, -1.0], [12.0, 0.25], [-6.0, 3.0]],
    [[-9.0, 5.5], [2.0, -2.0], [4.5, 6.0]],
    [[15.0, -0.5], [-11.0, 1.0], [3.0, -4.0]],
]
WINDOWS = {
    "左上角窗口 (0,0)-(2,2)": (0, 0, 2, 2),
    "右下角窗口 (3,2)-(4,3)": (3, 2, 4, 3),
    "整图窗口   (0,0)-(4,3)": (0, 0, 4, 3),
}


def run_offline() -> int:
    pack = pack_mod.build_pack(A_SCALE, B_SHIFT, KERNEL, BIAS)
    img = np.asarray(IMAGE, dtype=np.float64)
    ref = engine.reference_full_image(pack, img)
    return _verify(pack, img, ref, call=None)


def run_http(base_url: str) -> int:
    import httpx

    client = httpx.Client(base_url=base_url, timeout=10.0)
    r = client.get("/health")
    r.raise_for_status()
    print(f"[health] {r.json()}")

    r = client.post("/convert", json={
        "a": A_SCALE, "b": B_SHIFT, "kernel": KERNEL, "bias": BIAS,
    })
    r.raise_for_status()
    converted = r.json()
    print(f"[convert] pack_id={converted['pack_id']}  "
          f"C={converted['C']} O={converted['O']}  "
          f"A.shape={converted['folded_A_shape']} t.shape={converted['folded_t_shape']}")

    r = client.post("/reference", json={"pack": converted["pack"], "image": IMAGE})
    r.raise_for_status()
    ref = np.asarray(r.json()["output"], dtype=np.float64)

    def call(path: str, payload: dict) -> dict:
        resp = client.post(path, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        return resp.json()

    pack = pack_mod.decode_pack(converted["pack"])
    return _verify(
        pack, np.asarray(IMAGE, dtype=np.float64), ref,
        call=call, pack_token=converted["pack"], client=client,
    )


def _verify(pack, img, ref, call, pack_token=None, client=None) -> int:
    worst = 0.0
    for name, (r0, c0, r1, c1) in WINDOWS.items():
        window = {"r0": r0, "c0": c0, "r1": r1, "c1": c1}
        if pack_token is not None:
            coords_resp = call("/required-coordinates", {
                "pack": pack_token, "H": H, "W": W, "window": window,
            })
            coords = [tuple(x) for x in coords_resp["coords"]]
            active = coords_resp["active_channels"]
            pixels = {f"{rr},{cc}": list(img[rr, cc]) for rr, cc in coords}
            eval_resp = call("/evaluate", {
                "pack": pack_token, "H": H, "W": W,
                "window": window, "pixels": pixels,
            })
            out = np.asarray(eval_resp["output"], dtype=np.float64)
        else:
            info = engine.required_coordinates(pack, H, W, r0, c0, r1, c1)
            coords = [tuple(x) for x in info["coords"]]
            active = info["active_channels"]
            px = {(rr, cc): list(img[rr, cc]) for rr, cc in coords}
            out = engine.evaluate_window(pack, H, W, r0, c0, r1, c1, px)

        expected = ref[r0:r1, c0:c1]
        err = float(np.abs(out - expected).max())
        worst = max(worst, err)
        print(f"\n=== {name} ===")
        print(f"必读坐标 {len(coords)} 个: {coords}")
        print(f"活跃通道: {active}（通道 1 的 a=0，不读像素）")
        print(f"窗口响应:\n{np.array2string(out, precision=6, suppress_small=False)}")
        print(f"对照原模型最大绝对误差: {err:.3e}")
        assert err <= 1e-8, f"误差超界: {err}"

    print("\n[缺像素报错检查]")
    r0, c0, r1, c1 = WINDOWS["左上角窗口 (0,0)-(2,2)"]
    info = engine.required_coordinates(pack, H, W, r0, c0, r1, c1)
    coords = [tuple(x) for x in info["coords"]]
    if client is not None:
        import httpx
        bad_pixels = {f"{rr},{cc}": list(img[rr, cc]) for rr, cc in coords[1:]}
        resp = client.post("/evaluate", json={
            "pack": pack_token, "H": H, "W": W,
            "window": {"r0": r0, "c0": c0, "r1": r1, "c1": c1},
            "pixels": bad_pixels,
        })
        assert resp.status_code == 422, resp.text
        print(f"  HTTP 按预期返回 422: {resp.json()['detail']}")
    else:
        try:
            engine.evaluate_window(pack, H, W, r0, c0, r1, c1, pixels={})
        except engine.EngineError as exc:
            print(f"  离线引擎按预期报错: {exc}")
    print(f"\n全部窗口核对通过，最差误差 {worst:.3e} <= 1e-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="边角示例")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--offline", action="store_true", help="不起服务，直接调用引擎")
    args = ap.parse_args()
    if args.offline:
        return run_offline()
    try:
        return run_http(args.base_url)
    except Exception as exc:  # noqa: BLE001
        print(f"HTTP 模式失败（{exc}），可改用 --offline 直调引擎", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
