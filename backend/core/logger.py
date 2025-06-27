from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import logging
import time

# ----------------------------
# Configure logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ----------------------------
# Custom Logging Middleware
# ----------------------------
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        log_details = {
            "method": request.method,
            "url": request.url.path,
            "status_code": response.status_code,
            "duration": round(duration, 4)
        }

        logger.info(f"Request: {log_details}")
        return response