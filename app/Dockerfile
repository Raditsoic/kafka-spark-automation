FROM openjdk:21-slim

ENV SPARK_VERSION=3.5.3
ENV HADOOP_VERSION=3.4.0

# Install Python and dependencies
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y curl python3 python3-pip && \
    python3 --version \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt --break-system-packages

# Working Directory
WORKDIR /app

# Copy App files
COPY . /app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]