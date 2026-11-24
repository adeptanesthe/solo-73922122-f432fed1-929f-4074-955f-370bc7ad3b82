"""复核投票训练单示例：开单 -> 多质检员投票 -> 查单 -> 撤票改变梯度。

用法
----
# 1) 启动服务（Docker Compose 或本地）：
#    docker compose up --build
# 2) 运行示例：
python examples/training_demo.py                 # 默认 http://127.0.0.1:8000
python examples/training_demo.py --offline       # 不起服务，直接调引擎/训练单
python examples/training_demo.py --base-url http://其他地址:8000

场景
----
8x6 原始图、2 输入通道（通道 1 的 a=0，验证零通道仍求导）、2 输出通道。
三名质检员对若干输出点投缺陷/良品票，脚本逐轮打印有效点数、MSE 与
各输入通道 a、b 的梯度；最后用“撤回一张票”演示平票点被剔除后梯度如何变化，
并将服务返回值与离线全量重算结果逐值核对。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import engine, jobs as jobs_mod, pack as pack_mod  # noqa: E402

# ---------------------------------------------------------------- 模型与图
MODEL = {
    "a": [1.2, 0.0],          # 通道 1 的 a=0：不读像素，但梯度照算
    "b": [-0.5, 0.8],
    "kernel": [
        [[[0.5, -0.25, 0.0], [0.0, 1.0, 0.25], [0.5, 0.0, -0.5]],
         [[0.0, 0.25, 0.0], [-0.5, 0.0, 0.75], [0.0, 0.0, 0.5]]],
        [[[0.0, 0.0, 1.0], [-1.0, 0.5, 0.0], [0.0, 0.25, 0.0]],
         [[0.75, 0.0, -0.25], [0.0, 1.0, 0.0], [-0.5, 0.25, 0.0]]],
    ],
    "bias": [0.3, -0.7],
}
H, W = 8, 6
rng = np.random.default_rng(7)
IMAGE = rng.uniform(-8, 8, (H, W, 2)).tolist()

# 投票剧本：(质检员, [(r, c, o, label), ...])；空表表示撤回
SCRIPT = [
    ("alice", [(0, 0, 0, "defect"), (3, 2, 1, "defect"), (7, 5, 0, "good")]),
    ("bob",   [(0, 0, 0, "good"), (3, 2, 1, "good")]),
    ("carol", [(3, 2, 1, "good"), (5, 1, 1, "defect")]),
    # 演示撤票：撤回 bob 后，(0,0,0) 恢复为缺陷有效点，(3,2,1) 变为良品占优
    ("bob",   []),
]


def _items(entries):
    return [{"r": r, "c": c, "o": o, "label": label} for r, c, o, label in entries]


def _print_query(title, q, pack=None, img=None):
    print(f"\n--- {title} ---")
    if not q["has_effective_points"]:
        print(f"有效点数=0，提示：{q['message']}")
        return None
    print(f"有效点数 N={q['n_eff']}  MSE={q['mse']:.8f}")
    print(f"  grad_a = {np.array2string(np.asarray(q['grad_a']), precision=6)}")
    print(f"  grad_b = {np.array2string(np.asarray(q['grad_b']), precision=6)}")
    if pack is not None:
        # 离线全量重算对照（从当前票重建计票，不经增量累加器）
        return q
    return q


def run_offline() -> int:
    pack = pack_mod.build_pack(**MODEL)
    img = np.asarray(IMAGE, dtype=np.float64)
    job = jobs_mod.TrainingJob("job_demo", pack, img)

    print("== 离线模式：直接调用 TrainingJob ==")
    _print_query("开单初始状态", job.query())
    for inspector, entries in SCRIPT:
        ticket = jobs_mod.normalize_ticket(_items(entries), H, W, job.O)
        receipt = job.replace_ticket(inspector, ticket)
        print(f"\n[提交] inspector={inspector} action={receipt['action']} "
              f"票数={receipt['ticket_size']} 触及点={receipt['points_touched']}")
        _print_query(f"{inspector} 操作后查单", job.query())
        _check_against_full(job)

    # 撤回全部票 -> 无有效点提示
    for inspector in ("alice", "carol"):
        job.replace_ticket(inspector, {})
    _print_query("全部撤回后", job.query())
    print("\n离线演示完成：增量查单与全量重算全程一致。")
    return 0


def _check_against_full(job):
    check = job.full_recompute_check()
    stats = check["stats"]
    q = job.query()
    if stats["n_eff"] == 0:
        assert q["n_eff"] == 0
        return
    assert q["n_eff"] == stats["n_eff"]
    assert abs(q["mse"] - stats["mse"]) <= 1e-9
    np.testing.assert_allclose(q["grad_a"], stats["grad_a"], atol=1e-9)
    np.testing.assert_allclose(q["grad_b"], stats["grad_b"], atol=1e-9)


def run_http(base_url: str) -> int:
    import httpx

    client = httpx.Client(base_url=base_url, timeout=10.0)
    client.get("/health").raise_for_status()

    r = client.post("/convert", json=MODEL)
    r.raise_for_status()
    pack_id = r.json()["pack_id"]
    pack_token = r.json()["pack"]
    print(f"[convert] pack_id={pack_id}")

    r = client.post("/training-jobs", json={"pack_id": pack_id, "image": IMAGE})
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print(f"[open job] job_id={job_id}  H={r.json()['H']} W={r.json()['W']} "
          f"C={r.json()['C']} O={r.json()['O']}")

    def query():
        resp = client.post("/training-jobs/query", json={"job_id": job_id})
        assert resp.status_code == 200, resp.text
        return resp.json()

    _print_query("开单初始状态（无票）", query())

    pack = pack_mod.decode_pack(pack_token)
    img = np.asarray(IMAGE, dtype=np.float64)
    for inspector, entries in SCRIPT:
        resp = client.post("/training-jobs/ticket", json={
            "job_id": job_id, "inspector": inspector,
            "ticket": _items(entries),
        })
        assert resp.status_code == 200, resp.text
        receipt = resp.json()
        print(f"\n[提交] inspector={inspector} action={receipt['action']} "
              f"票数={receipt['ticket_size']} 触及点={receipt['points_touched']}")
        q = query()
        _print_query(f"{inspector} 操作后查单", q)
        # 与离线全量重算核对
        local = jobs_mod.TrainingJob("verify", pack, img)
        detail = client.get(f"/training-jobs/{job_id}").json()
        for t in detail["tickets"]:
            local.replace_ticket(
                t["inspector"],
                jobs_mod.normalize_ticket(t["ticket"], H, W, local.O),
            )
        ref = local.query()
        if ref["n_eff"]:
            assert abs(q["mse"] - ref["mse"]) <= 1e-9
            np.testing.assert_allclose(q["grad_a"], ref["grad_a"], atol=1e-9)
            np.testing.assert_allclose(q["grad_b"], ref["grad_b"], atol=1e-9)

    # 撤票改变梯度的直观演示：撤回 carol，(5,1,1) 失效、(3,2,1) 变为平票失效
    before = query()
    client.post("/training-jobs/ticket", json={
        "job_id": job_id, "inspector": "carol", "ticket": []})
    after = query()
    print("\n=== 撤票（carol 空表撤回）前后对比 ===")
    print(f"撤回前: N={before['n_eff']} MSE={before['mse']:.8f} "
          f"grad_a={np.array2string(np.asarray(before['grad_a']), precision=6)}")
    print(f"撤回后: N={after['n_eff']} MSE={after['mse']:.8f} "
          f"grad_a={np.array2string(np.asarray(after['grad_a']), precision=6)}")
    # 有效点数可能不变，但参与点的目标分/构成改变，MSE 与梯度必然随之变化
    assert after["mse"] != before["mse"]
    assert np.allclose(after["grad_a"], before["grad_a"]) is False

    # 全部撤回 -> 补标提示
    for inspector in ("alice",):
        client.post("/training-jobs/ticket", json={
            "job_id": job_id, "inspector": inspector, "ticket": []})
    final = query()
    print(f"\n全部撤回后: {final['message']}")
    assert final["has_effective_points"] is False
    print("\nHTTP 演示完成：服务查单与离线全量重算逐值一致，撤票即时改变梯度。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="复核投票训练单示例")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--offline", action="store_true", help="不起服务，直接调引擎")
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
