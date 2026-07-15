# 1. Usa uma imagem oficial do Python, versão slim para ficar leve
FROM python:3.12-slim

# 2. Define o diretório de trabalho lá dentro do container
WORKDIR /app

# 3. Instala dependências de sistema necessárias para compilar pacotes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Instala o gerenciador astral 'uv' que estamos usando no projeto
RUN pip install uv

# 5. Copia os arquivos de configuração do projeto para dentro do container
COPY pyproject.toml uv.lock README.md* ./

# 6. Instala o PyTorch versão CPU-only primeiro (economia drástica de ~8 GB!)
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu

# 7. Instala as demais dependências do backend e do frontend analítico
RUN uv pip install --system fastapi uvicorn loguru pandas mlflow streamlit plotly seaborn matplotlib pyyaml scipy scikit-learn

# 8. Copia o código-fonte da máquina para dentro do container
COPY src/ /app/src/

# 9. Expõe as portas que a API e o MLflow vão usar
EXPOSE 8000
EXPOSE 5000

# 10. Comando de disparo da API usando o servidor web Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]