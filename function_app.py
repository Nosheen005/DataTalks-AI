# function_app.py
import azure.functions as func

# If your package is laid out as src/<your_pkg>/api.py, adjust this import:
# e.g. from myproject.api import app as fastapi_app
from src.api import app as fastapi_app

# This exposes your FastAPI app as a single HTTP-triggered Azure Function
app = func.AsgiFunctionApp(
    app=fastapi_app,
    http_auth_level=func.AuthLevel.ANONYMOUS,
)
