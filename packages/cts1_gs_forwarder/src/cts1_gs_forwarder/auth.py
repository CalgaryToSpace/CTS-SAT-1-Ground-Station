from config import API_KEYS
from metrics import WS_AUTH_FAILURES_TOTAL


async def authenticate(session, dashboard_id: str, api_key: str) -> bool:
    expected = API_KEYS.get(dashboard_id)
    if not expected or expected != api_key:
        WS_AUTH_FAILURES_TOTAL.inc()
        return False

    session.dashboard_id = dashboard_id
    session.authenticated = True
    return True