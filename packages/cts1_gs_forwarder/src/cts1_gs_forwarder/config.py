import os
import logging
from dotenv import load_dotenv

load_dotenv()

API_KEYS = {
    "dashboard-alpha": os.getenv("DASHBOARD_ALPHA_API_KEY", "change-me-alpha"),
    "dashboard-beta": os.getenv("DASHBOARD_BETA_API_KEY", "change-me-beta"),
}

HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "15"))
HEARTBEAT_TIMEOUT_SECONDS = int(os.getenv("HEARTBEAT_TIMEOUT_SECONDS", "45"))
MAX_MESSAGE_SIZE_BYTES = int(os.getenv("MAX_MESSAGE_SIZE_BYTES", "65536"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("router")