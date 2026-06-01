# 1. Base Image
FROM python:3.9-slim

# 2. System Settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Work Directory: Establish the root folder inside the container
WORKDIR /app

# 4. Dependency Layer
COPY requirements.txt .

# 5. Strict Installation
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Source Code Layer
COPY . .

# 7. Execution
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]