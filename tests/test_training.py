"""训练单引擎测试：点梯度（有限差分）、增量计票与全量重算一致性、回滚、撤票。"""

from __future__ import annotations

import numpy as np
import pytest

from app import engine, jobs as jobs_mod, pack as pack_mod


def make_pack(C=2, O=3, seed=0, zero_a_channels=()):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-4, 4, C)
    for c in zero_a_channels:
        a[c] = 0.0
    b = rng.uniform(-3, 3, C)
    K = rng.uniform(-2, 2, (O, C, 3, 3))
    bias = rng.uniform(-1, 1, O)
    return pack_mod.build_pack(a, b, K, bias)


def point_value(pack, img, p, q, o):
    """用参考实现取单点值，作为 point_value_and_grad 的独立对照。"""
    return float(engine.reference_full_image(pack, img)[p, q, o])


# --------------------------------------------------------------------------- #
# 单点值与解析梯度
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(8))
def test_point_value_matches_reference(seed):
    rng = np.random.default_rng(seed)
    C, O = int(rng.integers(1, 5)), int(rng.integers(1, 5))
    H, W = int(rng.integers(1, 10)), int(rng.integers(1, 10))
    pack = make_pack(C=C, O=O, seed=seed,
                     zero_a_channels=(0,) if seed % 2 == 0 and C >= 2 else ())
    img = rng.uniform(-16, 16, (H, W, C))
    ref = engine.reference_full_image(pack, img)
    for p, q, o in [(0, 0, 0), (H - 1, W - 1, O - 1), (H // 2, W // 2, O // 2)]:
        z, dza, dzb = engine.point_value_and_grad(pack, img, p, q, o)
        assert z == pytest.approx(ref[p, q, o], abs=1e-12)
        assert dza.shape == (C,) and dzb.shape == (C,)


@pytest.mark.parametrize("seed", range(6))
def test_point_gradients_finite_difference(seed):
    """解析 ∂Z/∂a、∂Z/∂b 与中心有限差分一致（含 a=0 通道仍须有导数）。"""
    rng = np.random.default_rng(70 + seed)
    C, O = 3, 2
    H, W = 5, 6
    pack = make_pack(C=C, O=O, seed=seed, zero_a_channels=(1,))
    img = rng.uniform(-16, 16, (H, W, C))
    p, q, o = 2, 3, 1
    eps = 1e-6

    _, dza, dzb = engine.point_value_and_grad(pack, img, p, q, o)

    def perturbed(field, c, delta):
        """构造仅改单个 a/b 元素的参数包（数组深拷贝，避免共享污染）。"""
        new_pack = dict(pack)
        arr = np.asarray(pack[field], dtype=np.float64).copy()
        arr[c] += delta
        new_pack[field] = arr
        return new_pack

    for c in range(C):
        fd_a = (point_value(perturbed("a", c, eps), img, p, q, o)
                - point_value(perturbed("a", c, -eps), img, p, q, o)) / (2 * eps)
        assert dza[c] == pytest.approx(fd_a, rel=1e-6, abs=1e-6)

        fd_b = (point_value(perturbed("b", c, eps), img, p, q, o)
                - point_value(perturbed("b", c, -eps), img, p, q, o)) / (2 * eps)
        assert dzb[c] == pytest.approx(fd_b, rel=1e-6, abs=1e-6)

    # a=0 的通道 1：值不依赖其像素，但解析导数依然可能非零
    assert pack["a"][1] == 0.0
    _, dza1, _ = engine.point_value_and_grad(pack, img, p, q, o)
    assert np.any(dza1 != 0.0) or np.all(pack["K"][o, 1, :, :] == 0.0)


def test_point_grad_out_of_bounds_rejected():
    pack = make_pack()
    img = np.zeros((4, 4, 2))
    with pytest.raises(engine.EngineError):
        engine.point_value_and_grad(pack, img, 4, 0, 0)
    with pytest.raises(engine.EngineError):
        engine.point_value_and_grad(pack, img, 0, 0, 3)


# --------------------------------------------------------------------------- #
# 目标分规则
# --------------------------------------------------------------------------- #
def test_target_from_tally_rules():
    f = jobs_mod.TrainingJob.target_from_tally
    assert f(2, 1) == 1      # 缺陷票多 -> +1
    assert f(1, 2) == -1     # 良品票多 -> -1
    assert f(0, 0) is None   # 无票不参加
    assert f(2, 2) is None   # 平票不参加
    assert f(0, 3) == -1


# --------------------------------------------------------------------------- #
# 输入校验
# --------------------------------------------------------------------------- #
def test_validate_image_limits():
    pack = make_pack(C=2, O=2)
    jobs_mod.validate_image(np.zeros((64, 64, 2)), 2)  # 边界允许
    with pytest.raises(jobs_mod.JobError):
        jobs_mod.validate_image(np.zeros((65, 8, 2)), 2)
    with pytest.raises(jobs_mod.JobError):
        jobs_mod.validate_image(np.zeros((8, 65, 2)), 2)
    with pytest.raises(jobs_mod.JobError):
        jobs_mod.validate_image(np.zeros((8, 8, 3)), 2)  # 通道不符
    bad = np.zeros((4, 4, 2)); bad[0, 0, 0] = 16.5
    with pytest.raises(jobs_mod.JobError):
        jobs_mod.validate_image(bad, 2)


def test_normalize_ticket_validation():
    H, W, O = 4, 5, 3
    good = [{"r": 0, "c": 0, "o": 0, "label": "defect"},
            {"r": 3, "c": 4, "o": 2, "label": "good"}]
    out = jobs_mod.normalize_ticket(good, H, W, O)
    assert out == {(0, 0, 0): "defect", (3, 4, 2): "good"}
    assert jobs_mod.normalize_ticket([], H, W, O) == {}  # 空表=撤回
    with pytest.raises(jobs_mod.JobError):
        jobs_mod.normalize_ticket([{"r": 4, "c": 0, "o": 0, "label": "good"}], H, W, O)
    with pytest.raises(jobs_mod.JobError):
        jobs_mod.normalize_ticket([{"r": 0, "c": 0, "o": 3, "label": "good"}], H, W, O)
    with pytest.raises(jobs_mod.JobError):
        jobs_mod.normalize_ticket([{"r": 0, "c": 0, "o": 0, "label": "maybe"}], H, W, O)
    with pytest.raises(jobs_mod.JobError):  # 同人同点重复
        jobs_mod.normalize_ticket(
            [{"r": 1, "c": 1, "o": 0, "label": "defect"},
             {"r": 1, "c": 1, "o": 0, "label": "good"}], H, W, O)


# --------------------------------------------------------------------------- #
# 增量维护 vs 全量重算
# --------------------------------------------------------------------------- #
def _full_stats(job):
    check = job.full_recompute_check()
    return check["targets"], check["stats"]


def _assert_incremental_matches_full(job):
    targets, stats = _full_stats(job)
    assert job.snapshot_targets() == targets
    assert job._n_eff == stats["n_eff"]
    if stats["n_eff"] == 0:
        assert job.query()["n_eff"] == 0
        return
    q = job.query()
    assert q["n_eff"] == stats["n_eff"]
    # 增量路径的累加顺序与全量重算不同，容差按 float64 舍入量级放宽
    assert q["mse"] == pytest.approx(stats["mse"], rel=1e-11, abs=1e-10)
    np.testing.assert_allclose(q["grad_a"], stats["grad_a"], atol=1e-9, rtol=1e-11)
    np.testing.assert_allclose(q["grad_b"], stats["grad_b"], atol=1e-9, rtol=1e-11)
    # 内部累加量本身也应与全量分子和一致
    assert job._se_sum == pytest.approx(stats["se_sum"], rel=1e-11, abs=1e-10)
    np.testing.assert_allclose(job._ga_sum, stats["ga_sum"], atol=1e-9, rtol=1e-11)
    np.testing.assert_allclose(job._gb_sum, stats["gb_sum"], atol=1e-9, rtol=1e-11)


def test_empty_job_prompts_relabel():
    pack = make_pack(C=2, O=2, seed=3)
    img = np.random.default_rng(1).uniform(-16, 16, (6, 7, 2))
    job = jobs_mod.TrainingJob("job_x", pack, img)
    q = job.query()
    assert q["has_effective_points"] is False and q["n_eff"] == 0
    assert q["mse"] is None and q["grad_a"] is None and q["grad_b"] is None
    assert "补标" in q["message"]


def test_single_inspector_submit_replace_withdraw_incremental():
    pack = make_pack(C=2, O=2, seed=4)
    rng = np.random.default_rng(4)
    img = rng.uniform(-16, 16, (6, 6, 2))
    job = jobs_mod.TrainingJob("job_x", pack, img)

    t1 = jobs_mod.normalize_ticket(
        [{"r": 0, "c": 0, "o": 0, "label": "defect"},
         {"r": 2, "c": 3, "o": 1, "label": "good"},
         {"r": 5, "c": 5, "o": 0, "label": "defect"}], 6, 6, 2)
    r = job.replace_ticket("alice", t1)
    assert r["action"] == "submitted" and r["ticket_size"] == 3
    _assert_incremental_matches_full(job)

    # 整份替换：改标签、删点、加点都覆盖
    t2 = jobs_mod.normalize_ticket(
        [{"r": 0, "c": 0, "o": 0, "label": "good"},   # 翻转目标分
         {"r": 5, "c": 5, "o": 0, "label": "defect"},  # 保留
         {"r": 1, "c": 1, "o": 1, "label": "good"}],   # 新增；(2,3,1) 删除
        6, 6, 2)
    r = job.replace_ticket("alice", t2)
    assert r["action"] == "replaced" and r["points_touched"] == 4
    _assert_incremental_matches_full(job)

    # 空表撤回：所有点失效，回到无有效点状态
    r = job.replace_ticket("alice", {})
    assert r["action"] == "withdrawn" and r["ticket_size"] == 0
    assert job.snapshot_tickets() == {}
    _assert_incremental_matches_full(job)
    assert job.query()["has_effective_points"] is False


def test_multi_inspector_ties_and_majority():
    pack = make_pack(C=2, O=2, seed=5)
    rng = np.random.default_rng(5)
    img = rng.uniform(-16, 16, (5, 5, 2))
    job = jobs_mod.TrainingJob("job_x", pack, img)
    key = (1, 1, 0)

    # alice 缺陷票：有效点目标 +1
    job.replace_ticket("alice", jobs_mod.normalize_ticket(
        [{"r": 1, "c": 1, "o": 0, "label": "defect"}], 5, 5, 2))
    assert job.snapshot_targets() == {key: 1}
    _assert_incremental_matches_full(job)

    # bob 良品票：1:1 平票 -> 点失效
    job.replace_ticket("bob", jobs_mod.normalize_ticket(
        [{"r": 1, "c": 1, "o": 0, "label": "good"}], 5, 5, 2))
    assert job.snapshot_targets() == {}
    _assert_incremental_matches_full(job)

    # carol 再投良品：良品 2:1 -> 目标 -1
    job.replace_ticket("carol", jobs_mod.normalize_ticket(
        [{"r": 1, "c": 1, "o": 0, "label": "good"}], 5, 5, 2))
    assert job.snapshot_targets() == {key: -1}
    _assert_incremental_matches_full(job)

    # bob 撤回：缺陷 1:1 良品 -> 平票失效；只动 bob 票涉及的点
    job.replace_ticket("bob", {})
    assert job.snapshot_targets() == {}
    _assert_incremental_matches_full(job)

    # 全部撤回：计票字典应清空（无悬空键）
    job.replace_ticket("alice", {})
    job.replace_ticket("carol", {})
    assert job._tally == {} and job._n_eff == 0
    _assert_incremental_matches_full(job)


def test_random_ticket_churn_always_matches_full_recompute():
    """随机多轮提交/替换/撤回，增量累加量始终与全量重算逐值一致。"""
    rng = np.random.default_rng(2024)
    C, O = 3, 3
    pack = make_pack(C=C, O=O, seed=11, zero_a_channels=(2,))
    H, W = 8, 9
    img = rng.uniform(-16, 16, (H, W, C))
    job = jobs_mod.TrainingJob("job_rand", pack, img)
    inspectors = ["qa0", "qa1", "qa2", "qa3"]
    labels = (jobs_mod.LABEL_DEFECT, jobs_mod.LABEL_GOOD)

    for turn in range(60):
        inspector = inspectors[int(rng.integers(0, len(inspectors)))]
        if rng.random() < 0.2:
            ticket = {}  # 撤回
        else:
            size = int(rng.integers(0, 8))
            items = []
            seen = set()
            while len(items) < size:
                key = (int(rng.integers(0, H)), int(rng.integers(0, W)),
                       int(rng.integers(0, O)))
                if key in seen:
                    continue
                seen.add(key)
                items.append({"r": key[0], "c": key[1], "o": key[2],
                              "label": labels[int(rng.integers(0, 2))]})
            ticket = jobs_mod.normalize_ticket(items, H, W, O)
        job.replace_ticket(inspector, ticket)
        _assert_incremental_matches_full(job)


def test_failed_submission_keeps_old_ticket_and_stats():
    pack = make_pack(C=2, O=2, seed=6)
    rng = np.random.default_rng(6)
    img = rng.uniform(-16, 16, (5, 5, 2))
    job = jobs_mod.TrainingJob("job_x", pack, img)
    good_ticket = jobs_mod.normalize_ticket(
        [{"r": 0, "c": 0, "o": 0, "label": "defect"},
         {"r": 1, "c": 1, "o": 1, "label": "good"}], 5, 5, 2)
    job.replace_ticket("alice", good_ticket)
    before_tickets = job.snapshot_tickets()
    before_targets = job.snapshot_targets()
    before_n, before_se = job._n_eff, job._se_sum
    before_ga, before_gb = job._ga_sum.copy(), job._gb_sum.copy()
    before_receipt = job.last_receipt

    # 越界票：校验在归一化阶段失败，训练单状态完全不变
    with pytest.raises(jobs_mod.JobError):
        jobs_mod.normalize_ticket(
            [{"r": 9, "c": 0, "o": 0, "label": "defect"}], 5, 5, 2)

    # 空 inspector 触发 replace_ticket 内部回滚路径
    with pytest.raises(jobs_mod.JobError):
        job.replace_ticket("   ", good_ticket)
    assert job.snapshot_tickets() == before_tickets
    assert job.snapshot_targets() == before_targets
    assert job._n_eff == before_n and job._se_sum == before_se
    np.testing.assert_array_equal(job._ga_sum, before_ga)
    np.testing.assert_array_equal(job._gb_sum, before_gb)
    assert job.last_receipt == before_receipt


def test_gradient_formula_against_manual_loss_gradient():
    """用有限差分直接对 MSE 损失求导，核对查单梯度（含 a=0 通道）。"""
    C, O = 2, 2
    pack = make_pack(C=C, O=O, seed=8, zero_a_channels=(0,))
    rng = np.random.default_rng(8)
    H, W = 4, 4
    img = rng.uniform(-16, 16, (H, W, C))
    job = jobs_mod.TrainingJob("job_x", pack, img)
    items = [{"r": 0, "c": 0, "o": 0, "label": "defect"},
             {"r": 3, "c": 3, "o": 1, "label": "good"},
             {"r": 2, "c": 1, "o": 0, "label": "defect"}]
    job.replace_ticket("alice", jobs_mod.normalize_ticket(items, H, W, O))
    q = job.query()
    eps = 1e-6

    def loss_with(pack_override):
        check = job.full_recompute_check()
        stats = engine.training_stats_from_targets(
            pack_override, img, check["targets"])
        return stats["mse"]

    def perturbed(field, c, delta):
        new_pack = dict(pack)
        arr = np.asarray(pack[field], dtype=np.float64).copy()
        arr[c] += delta
        new_pack[field] = arr
        return new_pack

    for c in range(C):
        fd = (loss_with(perturbed("a", c, eps))
              - loss_with(perturbed("a", c, -eps))) / (2 * eps)
        assert q["grad_a"][c] == pytest.approx(fd, rel=1e-6, abs=1e-6)

        fd = (loss_with(perturbed("b", c, eps))
              - loss_with(perturbed("b", c, -eps))) / (2 * eps)
        assert q["grad_b"][c] == pytest.approx(fd, rel=1e-6, abs=1e-6)

    # a=0 通道（通道0）梯度同样被计算并通过有限差分
    assert pack["a"][0] == 0.0


def test_job_store_lru_and_unknown_id():
    pack = make_pack(C=1, O=1, seed=1)
    img = np.zeros((3, 3, 1))
    store = jobs_mod.JobStore(limit=3)
    ids = [store.create(pack, img).job_id for _ in range(4)]
    assert len(store) == 3
    with pytest.raises(jobs_mod.JobError):
        store.get(ids[0])  # 最早的被淘汰
    assert store.get(ids[-1]).job_id == ids[-1]
