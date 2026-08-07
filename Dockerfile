# syntax=docker/dockerfile:1

FROM python:3.14-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 의존성 레이어를 소스 코드와 분리해서 빌드 캐시 활용
COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY rag ./rag
COPY ui ./ui
COPY .streamlit ./.streamlit

# 영화 카탈로그/Chroma는 이미지에 넣지 않고 실행 시 읽기 전용으로 마운트한다.
# checkpoint와 UI 사용자 데이터만 별도 영구 볼륨에서 쓴다.
RUN mkdir -p /app/catalog /app/chroma_db /app/state /app/user_data

EXPOSE 8000 8501

CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
