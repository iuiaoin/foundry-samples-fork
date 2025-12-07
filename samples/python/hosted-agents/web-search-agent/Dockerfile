FROM python:3.12-slim

WORKDIR /app

# Copy wheel files first and install from local wheels
COPY build/*.whl /app/wheels/
RUN pip install /app/wheels/*.whl

COPY . user_agent/
WORKDIR /app/user_agent

# Install remaining dependencies from requirements.txt (excluding the wheel-installed package)
RUN if [ -f requirements.txt ]; then \
        grep -v "azure_ai_agentserver_agentframework" requirements.txt > requirements_filtered.txt && \
        pip install -r requirements_filtered.txt && \
        rm requirements_filtered.txt; \
    else \
        echo "No requirements.txt found"; \
    fi

EXPOSE 8088

CMD ["python", "main.py"]
