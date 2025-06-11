FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y git && \
    pip install --no-cache-dir pandas numpy sentence-transformers

# Copy files
COPY . .

# Run script
CMD ["python", "devops_ai_bot.py"]
