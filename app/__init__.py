"""边缘相机卷积转换 API。

模块划分：
- pack:   原始模型参数 -> 可复用参数包（折叠 a*x+b），base64 编解码
- engine: 必读坐标计算、稀疏原始像素窗口求值、原模型参考实现（仅校验用）
- schemas: Pydantic 请求/响应模型与约束校验
- main:   FastAPI 路由
"""

__all__ = ["pack", "engine", "schemas", "main"]
