FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000
# 单进程内存缓存参数包；多副本部署时请改用共享存储（参数包本身自包含）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
