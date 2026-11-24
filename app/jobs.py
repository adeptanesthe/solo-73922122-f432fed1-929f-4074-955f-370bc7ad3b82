"""质检员复核投票与标准化参数梯度训练单（进程内、线程安全、增量维护）。

业务规则
========
1. 一张训练单绑定一个现有卷积参数包与一张原始图（H、W 均 <= 64）。
2. 每位质检员可整份提交或整份替换自己的票；提交空表即撤回该人的全部票。
   同一人对同一输出点 ``(行 p, 列 q, 输出通道 o)`` 最多投一票。
3. 计票规则（按输出点汇总所有人的票）：
   - 缺陷票多于良品票 -> 目标分 ``+1``（有效点）；
   - 良品票多于缺陷票 -> 目标分 ``-1``（有效点）；
   - 平票或无票       -> 不参加训练（无效点）。
4. 改票为**增量**处理：只重算该质检员新旧票并集中涉及的输出点，
   不重扫其他人的票，也不重算整图。
5. 提交失败（任何校验/计算异常）时完整保留原票与此前的计算结果（快照回滚）。
6. 查单直接从已提交的票给出：有效点数、均方误差 MSE 以及各输入通道
   ``a``、``b`` 的梯度；无有效点则返回提示要求补标。

数学
====
原模型（整图补零规则不变）::

    Y[h,w,c] = a[c]*X[h,w,c] + b[c]，图外 tap 落在补零圈按 0 处理
    Z[p,q,o] = bias[o] + Σ_{图内 c,u,v} K[o,c,u,v] * Y[p+u,q+v,c]

有效点损失::

    L = (1/N) Σ_{有效点} (Z - target)^2，target ∈ {+1, -1}
    ∂L/∂a[c] = (2/N) Σ (Z-target) * ∂Z/∂a[c]
    ∂L/∂b[c] = (2/N) Σ (Z-target) * ∂Z/∂b[c]
其中 ``∂Z/∂a[c] = Σ_{图内u,v} K[o,c,u,v]*X[src,c]``、
``∂Z/∂b[c] = Σ_{图内u,v} K[o,c,u,v]``；导数式不含 a，故 ``a=0`` 通道仍须求导。

为 O(1) 查单，训练单长期维护跨有效点的四个累加量：有效点数 ``n_eff``、
平方误差和 ``se_sum``、梯度分子和 ``ga_sum/gb_sum``；改票时仅对受影响点
做“旧贡献扣减 / 新贡献加回”，与全量重算结果逐值一致（由测试核对）。
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import engine

# 票值 -> 目标分方向；defect=缺陷票，good=良品票
LABEL_DEFECT = "defect"
LABEL_GOOD = "good"
_LABEL_SIGN = {LABEL_DEFECT: 1, LABEL_GOOD: -1}
TARGET_DEFECT = 1     # 缺陷票多 -> 目标分 +1
TARGET_GOOD = -1      # 良品票多 -> 目标分 -1

MAX_SIDE = 64         # 原始图宽、高各至多 64 像素
MAX_ABS = 16.0        # 与 pack/engine/schemas 的统一数值约束一致
MAX_INSPECTOR_LEN = 64
_JOB_STORE_LIMIT = 64


class JobError(ValueError):
    """训练单/投票输入错误。"""


# --------------------------------------------------------------------------- #
# 输入校验
# --------------------------------------------------------------------------- #
def validate_image(image: np.ndarray, C: int) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 3:
        raise JobError(f"原图必须是 (H, W, C) 三维数组，收到形状 {img.shape}")
    H, W, img_c = img.shape
    if not (1 <= H <= MAX_SIDE and 1 <= W <= MAX_SIDE):
        raise JobError(f"原图高、宽必须在 1..{MAX_SIDE} 之间，收到 H={H}, W={W}")
    if img_c != C:
        raise JobError(f"原图通道数必须与参数包一致（C={C}），收到 {img_c}")
    if np.any(np.abs(img) > MAX_ABS):
        raise JobError(f"原始像素绝对值超过 {MAX_ABS:g}")
    return img


def normalize_ticket(
    ticket: Sequence[Mapping[str, Any]],
    H: int,
    W: int,
    O: int,
) -> Dict[Tuple[int, int, int], str]:
    """校验并归一化一个质检员的票；空表表示撤回。

    每条票形如 ``{"r": p, "c": q, "o": o, "label": "defect"|"good"}``。
    返回 ``{(p, q, o): label}``；同人同点重复即报错。
    """
    if not isinstance(ticket, (list, tuple)):
        raise JobError("ticket 必须是票条目数组（空数组表示撤回）")
    out: Dict[Tuple[int, int, int], str] = {}
    for idx, item in enumerate(ticket):
        if not isinstance(item, Mapping):
            raise JobError(f"第 {idx} 条票必须是对象")
        try:
            p, q, o = int(item["r"]), int(item["c"]), int(item["o"])
            label = str(item["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise JobError(
                f"第 {idx} 条票需包含整数 r、c、o 与字符串 label"
            ) from exc
        if not (0 <= p < H and 0 <= q < W):
            raise JobError(
                f"第 {idx} 条票坐标 (r={p}, c={q}) 越界，图尺寸 {H}x{W}"
            )
        if not 0 <= o < O:
            raise JobError(f"第 {idx} 条票输出通道 o={o} 越界，O={O}")
        if label not in _LABEL_SIGN:
            raise JobError(
                f"第 {idx} 条票 label 只能是 {LABEL_DEFECT!r} 或 {LABEL_GOOD!r}，"
                f"收到 {label!r}"
            )
        key = (p, q, o)
        if key in out:
            raise JobError(f"同一质检员对输出点 {key} 投了多票，每人每点至多一票")
        out[key] = label
    return out


# --------------------------------------------------------------------------- #
# 训练单
# --------------------------------------------------------------------------- #
class TrainingJob:
    """绑定参数包与原始图的投票训练单（所有可变状态均在锁内访问）。"""

    def __init__(self, job_id: str, pack: Mapping[str, np.ndarray],
                 image: np.ndarray) -> None:
        self.job_id = job_id
        self._pack = dict(pack)
        self._image = np.asarray(image, dtype=np.float64).copy()
        self.H, self.W, self.C = (int(v) for v in self._image.shape)
        self.O = int(self._pack["O"])

        # inspector -> {(p,q,o): label}，已提交的票
        self._tickets: Dict[str, Dict[Tuple[int, int, int], str]] = {}
        # 输出点 -> [缺陷票数, 良品票数]
        self._tally: Dict[Tuple[int, int, int], List[int]] = {}
        # 有效输出点 -> 目标分 (+1/-1)
        self._targets: Dict[Tuple[int, int, int], int] = {}
        # 跨有效点累加量（增量维护）
        self._n_eff = 0
        self._se_sum = 0.0
        self._ga_sum = np.zeros(self.C, dtype=np.float64)
        self._gb_sum = np.zeros(self.C, dtype=np.float64)
        self._lock = threading.RLock()
        # 最近一次提交回执（失败时不变）
        self.last_receipt: Optional[Dict[str, Any]] = None

    # ----- 只读视图（加锁拷贝，避免外部变更内部状态）----- #
    def snapshot_tickets(self) -> Dict[str, Dict[Tuple[int, int, int], str]]:
        with self._lock:
            return {k: dict(v) for k, v in self._tickets.items()}

    def snapshot_targets(self) -> Dict[Tuple[int, int, int], int]:
        with self._lock:
            return dict(self._targets)

    def inspectors(self) -> List[str]:
        with self._lock:
            return sorted(self._tickets)

    # ----- 目标分：纯函数，便于单测 ----- #
    @staticmethod
    def target_from_tally(n_defect: int, n_good: int) -> Optional[int]:
        if n_defect > n_good:
            return TARGET_DEFECT
        if n_good > n_defect:
            return TARGET_GOOD
        return None  # 平票或无票：不参加训练

    # ----- 单点对累加量的贡献 ----- #
    def _point_contribution(
        self, key: Tuple[int, int, int], target: int
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        p, q, o = key
        z, dza, dzb = engine.point_value_and_grad(
            self._pack, self._image, p, q, o
        )
        resid = z - float(target)
        return resid * resid, resid * dza, resid * dzb

    def _remove_contribution(self, key: Tuple[int, int, int]) -> None:
        """有效点退出训练集（或目标分改变前）：扣减其旧贡献。"""
        target = self._targets.pop(key)
        se, ga, gb = self._point_contribution(key, target)
        self._n_eff -= 1
        self._se_sum -= se
        self._ga_sum -= ga
        self._gb_sum -= gb

    def _add_contribution(self, key: Tuple[int, int, int], target: int) -> None:
        """有效点进入训练集（或目标分改变后）：加回其新贡献。"""
        self._targets[key] = target
        se, ga, gb = self._point_contribution(key, target)
        self._n_eff += 1
        self._se_sum += se
        self._ga_sum += ga
        self._gb_sum += gb

    def _recompute_point(self, key: Tuple[int, int, int]) -> None:
        """根据当前计票刷新单个输出点的有效性/目标分，并增量调整累加量。"""
        n_defect, n_good = self._tally.get(key, [0, 0])
        new_target = self.target_from_tally(n_defect, n_good)
        old_target = self._targets.get(key)
        if new_target == old_target:
            return  # 有效性与目标分均未变（含“仍无效”）：无需触碰累加量
        if old_target is not None:
            self._remove_contribution(key)
        if new_target is not None:
            self._add_contribution(key, new_target)
        # 已无任何票的点清理计票字典，控制内存随票数而非图规模增长
        if key in self._tally and n_defect == 0 and n_good == 0:
            del self._tally[key]

    # ----- 提交 / 替换 / 撤回 ----- #
    def replace_ticket(
        self, inspector: str, new_ticket: Mapping[Tuple[int, int, int], str]
    ) -> Dict[str, Any]:
        """整份替换某质检员的票（空表即撤回），增量刷新受影响点。

        任何异常都先回滚到调用前状态，原票与计算结果保持不变。
        """
        if not isinstance(inspector, str) or not inspector.strip():
            raise JobError("inspector 不能为空")
        if len(inspector) > MAX_INSPECTOR_LEN:
            raise JobError(f"inspector 长度不能超过 {MAX_INSPECTOR_LEN}")
        new_ticket = dict(new_ticket)

        with self._lock:
            # 失败回滚快照：仅覆盖本方法可能改动的状态（嵌套列表原样保留，
            # 回滚时恢复同一对象，保证与快照逐对象一致）
            snap = (
                dict(self._tickets),
                dict(self._tally),
                dict(self._targets),
                self._n_eff,
                self._se_sum,
                self._ga_sum.copy(),
                self._gb_sum.copy(),
                self.last_receipt,
            )
            old_ticket = dict(self._tickets.get(inspector, {}))
            try:
                if inspector in self._tickets:
                    del self._tickets[inspector]
                if new_ticket:
                    self._tickets[inspector] = dict(new_ticket)

                # 只处理该质检员新旧票并集涉及的点，其他人的票原样保留
                affected = set(old_ticket) | set(new_ticket)
                touched: List[Tuple[int, int, int]] = []
                for key in sorted(affected):
                    old_label = old_ticket.get(key)
                    new_label = new_ticket.get(key)
                    if old_label == new_label:
                        touched.append(key)
                        continue  # 该点本人票未变，计票不变
                    if old_label is not None:
                        cell = self._tally.setdefault(key, [0, 0])
                        cell[0 if old_label == LABEL_DEFECT else 1] -= 1
                    if new_label is not None:
                        cell = self._tally.setdefault(key, [0, 0])
                        cell[0 if new_label == LABEL_DEFECT else 1] += 1
                    self._recompute_point(key)
                    touched.append(key)

                receipt = {
                    "job_id": self.job_id,
                    "inspector": inspector,
                    "action": "withdrawn" if not new_ticket else (
                        "replaced" if old_ticket else "submitted"
                    ),
                    "ticket_size": len(new_ticket),
                    "points_touched": len(touched),
                    "n_eff": self._n_eff,
                }
                self.last_receipt = receipt
                return receipt
            except Exception:
                (
                    self._tickets,
                    self._tally,
                    self._targets,
                    self._n_eff,
                    self._se_sum,
                    self._ga_sum,
                    self._gb_sum,
                    self.last_receipt,
                ) = snap
                raise

    # ----- 查单 ----- #
    def query(self) -> Dict[str, Any]:
        """从已提交的票计算有效点数、MSE 与各通道 a/b 梯度。"""
        with self._lock:
            if self._n_eff == 0:
                return {
                    "job_id": self.job_id,
                    "has_effective_points": False,
                    "n_eff": 0,
                    "mse": None,
                    "grad_a": None,
                    "grad_b": None,
                    "message": "当前没有有效复核点（平票或无票不参加训练），请质检员补标后再查单。",
                }
            n = float(self._n_eff)
            return {
                "job_id": self.job_id,
                "has_effective_points": True,
                "n_eff": self._n_eff,
                "mse": self._se_sum / n,
                # MSE = (1/N)Σres² => 梯度带 2/N 系数
                "grad_a": ((2.0 / n) * self._ga_sum).tolist(),
                "grad_b": ((2.0 / n) * self._gb_sum).tolist(),
                "message": "ok",
            }

    def full_recompute_check(self) -> Dict[str, Any]:
        """全量重算核对：从当前所有票重建计票并逐点计算，供测试/自检使用。"""
        with self._lock:
            tally: Dict[Tuple[int, int, int], List[int]] = {}
            for ticket in self._tickets.values():
                for key, label in ticket.items():
                    cell = tally.setdefault(key, [0, 0])
                    cell[0 if label == LABEL_DEFECT else 1] += 1
            targets = {
                key: tgt
                for key, (nd, ng) in tally.items()
                if (tgt := self.target_from_tally(nd, ng)) is not None
            }
            stats = engine.training_stats_from_targets(
                self._pack, self._image, targets
            )
            return {
                "tally": tally,
                "targets": targets,
                "stats": stats,
            }


# --------------------------------------------------------------------------- #
# 训练单存储（进程内 LRU，线程安全）
# --------------------------------------------------------------------------- #
class JobStore:
    def __init__(self, limit: int = _JOB_STORE_LIMIT) -> None:
        self._items: "OrderedDict[str, TrainingJob]" = OrderedDict()
        self._limit = limit
        self._lock = threading.RLock()

    def create(self, pack: Mapping[str, np.ndarray], image: np.ndarray) -> TrainingJob:
        with self._lock:
            job = TrainingJob(
                "job_" + uuid.uuid4().hex[:16], pack, image
            )
            self._items[job.job_id] = job
            self._items.move_to_end(job.job_id)
            while len(self._items) > self._limit:
                self._items.popitem(last=False)
            return job

    def get(self, job_id: str) -> TrainingJob:
        with self._lock:
            try:
                job = self._items[job_id]
                self._items.move_to_end(job_id)
            except KeyError:
                raise JobError(f"未知 job_id: {job_id}") from None
            return job

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
