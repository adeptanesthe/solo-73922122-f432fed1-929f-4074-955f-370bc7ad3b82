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
| GET  | `/health` | 健康检查 |

窗口为半开区间 `[r0,r1) × [c0,c1)`，输出坐标。约束：输入/输出通道各 1..4；
像素与模型参数绝对值 ≤ 16；`a` 允许为 0。

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

## 目录

```
app/            FastAPI 应用
  pack.py       参数包折叠（A=K·a, t=K·b）、base64 编解码、LRU 存储
  engine.py     必读坐标、稀疏原始像素求值、原模型参考实现
  schemas.py    Pydantic 约束校验
  main.py       路由
examples/corner_demo.py   边角示例（HTTP / 离线）
tests/          29 项单元与端到端测试
```

## 已知限制 / 剩余工作

- 参数包缓存为进程内 LRU；多副本/重启场景应换 Redis 等共享存储，或直接传自包含 `pack`。
- 参考实现 `/reference` 会显式构造标准化块，仅用于核对，不参与部署路径。
- 当前按请求即时 NumPy 运算；超大窗口批量场景可进一步加 BLAS 批处理与限流。
