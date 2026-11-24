# 边缘相机卷积转换 API

面向边缘相机按窗口读取原图的缺陷检测部署场景：把“逐像素标准化 + 外圈补零 + 3×3 互相关”
的原模型，转换成**直接吃原始像素**的参数包，并精确列出每个窗口在数学上必读的图内坐标。
全程不生成标准化像素块，参数包体积与整图/窗口尺寸无关。

## 数学背景

原模型（`X` 为 H×W×C 原图，float64）：

```
Y[h,w,c] = a[c]·X[h,w,c] + b[c]          # 图内逐像素标准化
Y_pad    = Y 外圈补一圈零
Z[p,q,o] = bias[o] + Σ_{c,u,v} K[o,c,u,v]·Y_pad[p+u, q+v, c]   # 步长 1 互相关
```

把标准化展开并按“源像素是否在图内”拆分，得到部署公式（`A = K·a`，`t = K·b`）：

```
Z[p,q,o] = bias[o]
         + Σ_{图内 tap} A[o,c,u,v]·X[源r,源c,c]   # 原始像素直接乘，无标准化块
         + Σ_{图内 tap} t[o,c,u,v]                # 常量按有效位掩码逐点修正
```

- 补零圈位置两个求和都不计入，由 `(源r,源c)` 的图内有效掩码处理，边角/单行单列自动正确；
- `a[c]=0` 时该通道不读取任何像素（`A` 整列为 0），但 `t=K·b` 的常量贡献仍然保留；
- 参数包只含 `(O,C,3,3)` 及以下尺寸的数组，换图、换窗口直接复用。

## 必读坐标的精确裁剪

某图内像素必读，当且仅当同时满足：

1. 通道 `c` 满足 `a[c] ≠ 0` 且该 tap 在某个输出通道下权重 `K[o,c,u,v] ≠ 0`；
2. 源点 `(p+u, q+v)` 落在图内；
3. 对应输出点 `(p,q)` 落在请求窗口内。

零权重 tap、`a=0` 通道、只落在补零圈的位置一律不读。`/evaluate` 只接收这些原始像素；
缺任一必读坐标返回 HTTP 422。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/convert` | 传 `a/b/kernel/bias`，返回 base64 参数包 `pack` 与缓存 `pack_id` |
| POST | `/required-coordinates` | 传参数包、整图 H×W、窗口，返回必读坐标（含分通道列表） |
| POST | `/evaluate` | 传参数包、尺寸、窗口、原始像素字典，返回窗口响应 |
| POST | `/reference` | 原模型整图实现（显式标准化+补零），仅用于误差核对 |
| POST | `/training-jobs` | 用现有参数包 + 原始图（H、W 各 ≤ 64）开复核训练单 |
| POST | `/training-jobs/ticket` | 质检员整份提交/替换自己的票，空表撤回 |
| POST | `/training-jobs/query` | 查单：有效点数、MSE、各输入通道 a/b 梯度 |
| GET  | `/training-jobs/{job_id}` | 查看训练单上各质检员当前已提交的票 |
| GET  | `/health` | 健康检查 |

窗口为半开区间 `[r0,r1) × [c0,c1)`，输出坐标。约束：输入/输出通道各 1..4；
像素与模型参数绝对值 ≤ 16；`a` 允许为 0。

## 质检员复核投票与标准化参数梯度

相机换灯后，训练脚本改用**质检员复核过的输出点**计算标准化参数 `a`、`b` 的梯度。

### 投票规则

- 一张训练单绑定一个现有卷积参数包与一张原始图（高、宽各 1..64 像素）；
- 每位质检员可**整份提交或整份替换**自己的票，提交**空表即撤回**；
- 每人对同一输出点 `(行 r, 列 c, 输出通道 o)` 最多投一票，标签
  `defect`（缺陷）/ `good`（良品）；
- 按输出点汇总所有人的票：缺陷票多 → 目标分 **+1**，良品票多 → 目标分 **−1**，
  **平票或无票的点不参加训练**；
- 改票只增量处理该质检员新旧票并集涉及的点，不重扫其他人的票，也不重算整图；
- 提交失败（越界坐标、非法标签、同人同点重复等）返回 HTTP 422，
  服务端快照回滚，**原票与此前计算结果完整保留**。

### 梯度定义（沿用原模型整图补零规则）

有效点集合（N 个）上的均方误差与梯度：

```
L        = (1/N) Σ_{有效 (p,q,o)} (Z[p,q,o] - target)²
∂L/∂a[c] = (2/N) Σ (Z-target) · ∂Z/∂a[c]
∂L/∂b[c] = (2/N) Σ (Z-target) · ∂Z/∂b[c]
∂Z/∂a[c] = Σ_{图内 u,v} K[o,c,u,v] · X[p+u,q+v,c]
∂Z/∂b[c] = Σ_{图内 u,v} K[o,c,u,v]
```

图外 tap 落在补零圈按 0 处理，对值与导数均无贡献（图内掩码自动排除）。
导数式本身不含 `a`，所以 **`a[c]=0` 的通道仍照常求导**（测试以有限差分核对）。
训练单长期维护 `N`、平方误差和、两个梯度分子和四个累加量，改票时仅对受影响点
做旧贡献扣减/新贡献加回，查单为 O(1)；增量结果与全量重算逐值一致（见
`tests/test_training.py` 的 60 轮随机改票核对）。无有效点时查单返回
`has_effective_points=false` 与补标提示，`mse/grad_a/grad_b` 为 `null`。

### curl 最小示例（撤票怎样改变梯度）

```bash
# 1) 转换参数包（通道 1 的 a=0，其梯度仍会计算）
curl -s -X POST http://127.0.0.1:8000/convert -H 'Content-Type: application/json' -d '{
  "a": [1.2, 0.0], "b": [-0.5, 0.8],
  "kernel": [
    [[[0.5,-0.25,0],[0,1,0.25],[0.5,0,-0.5]],[[0,0.25,0],[-0.5,0,0.75],[0,0,0.5]]],
    [[[0,0,1],[-1,0.5,0],[0,0.25,0]],[[0.75,0,-0.25],[0,1,0],[-0.5,0.25,0]]]
  ],
  "bias": [0.3, -0.7]
}'
# -> {"pack":"<base64>","pack_id":"pack_...", ...}

# 2) 开单（4x3 原图，宽高均 <= 64）
curl -s -X POST http://127.0.0.1:8000/training-jobs -H 'Content-Type: application/json' -d '{
  "pack_id": "pack_...",
  "image": [[[1,2],[-3,4],[7,-8]],[[0.5,-1],[12,0.25],[-6,3]],
            [[-9,5.5],[2,-2],[4.5,6]],[[15,-0.5],[-11,1],[3,-4]]]
}'
# -> {"job_id":"job_...","H":4,"W":3,"C":2,"O":2,...}

# 3) 无票查单 -> 提示补标
curl -s -X POST http://127.0.0.1:8000/training-jobs/query \
  -H 'Content-Type: application/json' -d '{"job_id":"job_..."}'
# -> {"has_effective_points":false,"n_eff":0,"mse":null,...,"message":"...请质检员补标..."}

# 4) alice 整份投票：(0,0,0) 缺陷、(1,1,1) 良品
curl -s -X POST http://127.0.0.1:8000/training-jobs/ticket \
  -H 'Content-Type: application/json' -d '{"job_id":"job_...","inspector":"alice",
  "ticket":[{"r":0,"c":0,"o":0,"label":"defect"},
            {"r":1,"c":1,"o":1,"label":"good"}]}'
# -> {"action":"submitted","ticket_size":2,"points_touched":2,"n_eff":2}
# 查单 -> {"n_eff":2,"mse":21.29,"grad_a":[40.17,2.39],"grad_b":[-6.12,-10.20],...}

# 5) bob 对 (0,0,0) 投良品 -> 1:1 平票，该点被剔除，只剩 1 个有效点，梯度变化
curl -s -X POST http://127.0.0.1:8000/training-jobs/ticket \
  -H 'Content-Type: application/json' -d '{"job_id":"job_...","inspector":"bob",
  "ticket":[{"r":0,"c":0,"o":0,"label":"good"}]}'
# 查单 -> {"n_eff":1,"mse":38.86,"grad_a":[74.37,-21.13],"grad_b":[-9.35,-15.58],...}

# 6) bob 交空表撤回 -> 平票解除，(0,0,0) 恢复有效，梯度精确回到第 4 步
curl -s -X POST http://127.0.0.1:8000/training-jobs/ticket \
  -H 'Content-Type: application/json' -d '{"job_id":"job_...","inspector":"bob","ticket":[]}'
# -> {"action":"withdrawn","ticket_size":0,"points_touched":1,"n_eff":2}
# 查单 -> {"n_eff":2,"mse":21.29,"grad_a":[40.17,2.39],"grad_b":[-6.12,-10.20],...}
```

可直接运行的完整剧本（含离线全量重算核对）：

```bash
python examples/training_demo.py             # 请求运行中的服务
python examples/training_demo.py --offline   # 不起服务，直接调引擎
```

## 启动

### Docker Compose

```bash
docker compose up --build
# 文档页 http://127.0.0.1:8000/docs
```

### 本地（无 Docker 时）

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 可调用的边角示例

脚本同时支持在线 HTTP 与离线直调：

```bash
python examples/corner_demo.py             # 请求运行中的服务
python examples/corner_demo.py --offline   # 不起服务，直接调引擎
```

场景：4×3 图、2 输入通道（通道 1 的 `a=0`）、3 输出通道；取左上角、右下角、整图三个
补零最密集的窗口，与 `/reference` 原模型整图结果逐值比对，并演示缺像素 422。

### curl 最小示例（右下角 1×1 边角窗口）

```bash
# 1) 转换参数包
curl -s -X POST http://127.0.0.1:8000/convert -H 'Content-Type: application/json' -d '{
  "a": [1.5, 0.0], "b": [-2.0, 0.75],
  "kernel": [
    [[[0,0,0],[0,1,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]],
    [[[0.25,0,-0.5],[0,1,0],[0.5,0,-0.25]], [[0,0,0],[0,0,0],[0,0,0]]],
    [[[0,0,0],[0,0,0],[0,0,0]], [[-1.25,0,0],[0,0,0],[0,0,0]]]
  ],
  "bias": [0.5, -1.0, 2.0]
}'
# -> {"pack": "<base64>", "pack_id": "pack_...", "C": 2, "O": 3, ...}

# 2) 必读图内坐标（右下角窗口只需 2 个像素）
curl -s -X POST http://127.0.0.1:8000/required-coordinates \
  -H 'Content-Type: application/json' \
  -d '{"pack":"<base64>","H":4,"W":3,
       "window":{"r0":3,"c0":2,"r1":4,"c1":3}}'
# -> {"coords":[[2,1],[3,2]], "active_channels":[0], "count":2, ...}

# 3) 用原始像素求值（未标准化；缺坐标返回 422）
curl -s -X POST http://127.0.0.1:8000/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"pack":"<base64>","H":4,"W":3,
       "window":{"r0":3,"c0":2,"r1":4,"c1":3},
       "pixels":{"2,1":[2.0,-2.0],"3,2":[3.0,-4.0]}}'
# -> {"shape":[1,1,3],"output":[[[3.0,1.75,1.0625]]],"pixels_used":2}
```

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

测试覆盖：原转换链路 29 项；训练单引擎 25 项（点值对照参考实现、解析梯度的
有限差分核对含 `a=0` 通道、目标分规则、60 轮随机改票的增量/全量一致性、
失败回滚、LRU 存储）；训练 API 端到端 7 项（开单约束、投票生命周期、
平票剔除与多数翻转、撤票改梯度、非法票 422 不留痕）。

## 目录

```
app/            FastAPI 应用
  pack.py       参数包折叠（A=K·a, t=K·b）、base64 编解码、LRU 存储
  engine.py     必读坐标、稀疏原始像素求值、原模型参考实现、
                输出点值与 a/b 解析梯度、全量训练统计（增量核对基准）
  jobs.py       复核训练单：增量计票、有效点目标分、MSE/梯度累加量、
                快照回滚、线程安全 LRU 训练单存储
  schemas.py    Pydantic 约束校验
  main.py       路由
examples/corner_demo.py     边角示例（HTTP / 离线）
examples/training_demo.py   复核投票/撤票改变梯度示例（HTTP / 离线）
tests/          61 项单元与端到端测试
```

## 已知限制 / 剩余工作

- 参数包缓存与训练单均为进程内 LRU；多副本/重启场景应换 Redis 等共享存储，
  或直接传自包含 `pack`（训练单本身不支持跨进程迁移）。
- 参考实现 `/reference` 会显式构造标准化块，仅用于核对，不参与部署路径。
- 当前按请求即时 NumPy 运算；超大窗口批量场景可进一步加 BLAS 批处理与限流。
- 训练单存储有上限（默认 64 张），超限按 LRU 淘汰；无鉴权/配额，
  质检员身份由请求体 `inspector` 自报，生产部署需在网关层补认证。
