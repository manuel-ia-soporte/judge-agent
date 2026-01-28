import httpx
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--agent-url",
        default="",
        help="Running agent base URL (e.g. http://localhost:9009). If omitted, HTTP conformance tests are skipped.",
    )


@pytest.fixture(scope="session")
def agent_url(request):
    url = str(request.config.getoption("--agent-url") or "").rstrip("/")
    if not url:
        pytest.skip("No --agent-url provided; skipping HTTP conformance tests.")

    try:
        response = httpx.get(f"{url}/.well-known/agent-card.json", timeout=5)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Could not connect to agent at {url}: {exc}")

    return url
