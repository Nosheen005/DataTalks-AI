FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy requirements (or pyproject)
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . /app

# Streamlit needs these (otherwise it shows "sharing" menu etc.)
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8502
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose Streamlit port
EXPOSE 8502

# Run Streamlit app
CMD ["streamlit", "run", "src/streamlit_app.py"]
