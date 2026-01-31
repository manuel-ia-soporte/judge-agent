# AgentBeats Finance Competition - Project Context

## Overview

This project consists of **3 repositories** that together form a financial agent evaluation platform for the **AgentBeats Competition (Phase 1)**.

```
agentbeats-competition/
├── GreenAgentFinance.git/           # Evaluator agent (judge)
├── purple-agent-finance.git/        # Participant agent (competitor)
└── GreenAgentFinance-Leaderboard.git/  # Orchestrator (CI/CD + Docker Compose)
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EVALUATION FLOW                                │
│                                                                             │
│   Participant        GreenAgentFinance-Leaderboard         Docker Network   │
│       │                        │                                │           │
│       │  Fork + Edit           │                                │           │
│       │  scenario.toml         │                                │           │
│       │───────────────────────>│                                │           │
│       │                        │  GitHub Actions                │           │
│       │                        │  docker compose up             │           │
│       │                        │───────────────────────────────>│           │
│       │                        │                                │           │
│       │                        │     ┌──────────────────────────┴─────┐    │
│       │                        │     │  ┌─────────────────────────┐   │    │
│       │                        │     │  │  AgentBeats Client      │   │    │
│       │                        │     │  │  (Orchestrator)         │   │    │
│       │                        │     │  └───────────┬─────────────┘   │    │
│       │                        │     │              │                 │    │
│       │                        │     │              │ A2A Protocol    │    │
│       │                        │     │              ▼                 │    │
│       │                        │     │  ┌─────────────────────────┐   │    │
│       │                        │     │  │  Green Agent :9009      │   │    │
│       │                        │     │  │  (Evaluator)            │   │    │
│       │                        │     │  └───────────┬─────────────┘   │    │
│       │                        │     │              │                 │    │
│       │                        │     │              │ Questions       │    │
│       │                        │     │              │ (public.csv)    │    │
│       │                        │     │              ▼                 │    │
│       │                        │     │  ┌─────────────────────────┐   │    │
│       │                        │     │  │  Purple Agent :9009     │   │    │
│       │                        │     │  │  (Participant)          │   │    │
│       │                        │     │  └─────────────────────────┘   │    │
│       │                        │     └────────────────────────────────┘    │
│       │                        │                                │           │
│       │                        │<───────────────────────────────│           │
│       │                        │         results.json           │           │
│       │                        │                                │           │
│       │  PR with results       │                                │           │
│       │<───────────────────────│                                │           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Details

### 1. GreenAgentFinance (Evaluator Agent)

**Purpose:** Evaluates participant agents against a financial Q&A dataset using offline tools and rubric-based scoring.

**Tech Stack:**
- Python 3.13, FastAPI, Uvicorn
- Custom A2A protocol implementation
- model-library for LLM abstraction

**Key Components:**
```
src/finance_green_agent/
├── server.py          # FastAPI A2A endpoints
├── green_eval.py      # Evaluation orchestration
├── a2a_schemas.py     # A2A protocol data models
├── task_store.py      # In-memory task storage
├── agent_core/        # Core agent logic (agent.py, prompt.py, tools_base.py)
├── tools/             # OFFLINE tools (web_search, edgar_search, html_parser)
└── eval/              # Scoring (rubric.py, public_eval.py)
```

**Tools (OFFLINE - uses local cache):**
| Tool | Description |
|------|-------------|
| `offline_web_search` | Search pre-downloaded web pages |
| `offline_edgar_search` | Search pre-downloaded SEC filings |
| `parse_cached_html` | Extract text from cached HTML |
| `citation_validator` | Validate source citations |

**Docker Image:** `ghcr.io/elvlandau117/greenagentfinance`

---

### 2. purple-agent-finance (Participant Agent)

**Purpose:** Answers financial questions using online tools and LLM reasoning. This is the agent being evaluated.

**Tech Stack:**
- Python 3.13, a2a-sdk, Starlette, Uvicorn
- model-library for LLM (default: gpt-4o-mini)

**Key Components:**
```
src/
├── server.py                    # A2A server setup
├── executor.py                  # Request handler
├── agent.py                     # Main agent implementation
└── purple_agent_finance/
    ├── finance_agent.py         # Core agent with tool-calling loop
    ├── tools.py                 # Tool definitions
    ├── prompt.py                # System prompts
    ├── get_agent.py             # Agent factory (detects OpenRouter models)
    └── openrouter_provider.py   # OpenRouter LLM provider wrapper
```

**Tools (ONLINE - live API calls):**
| Tool | API | Description |
|------|-----|-------------|
| `google_web_search` | SerpAPI | Live web search |
| `edgar_search` | SEC API | Live SEC EDGAR filings search |
| `parse_html_page` | Web scraping | Fetch and parse HTML from URLs |
| `retrieve_information` | LLM | RAG over parsed documents |

**Environment Variables:**
```bash
SERPAPI_API_KEY=xxx          # Required for web search
SEC_EDGAR_API_KEY=xxx        # Required for SEC EDGAR
FINANCE_MODEL=openai/gpt-4o-mini  # Or openrouter/openai/gpt-4o for OpenRouter
FINANCE_MAX_TURNS=8
FINANCE_TEMPERATURE=0
OPENROUTER_API_KEY=xxx       # Required only for OpenRouter models
```

**Docker Image:** `ghcr.io/elvlandau117/purple-agent-finance`

---

### 3. GreenAgentFinance-Leaderboard (Orchestrator)

**Purpose:** Automates evaluation runs via GitHub Actions and Docker Compose.

**Tech Stack:**
- Python 3.11, GitHub Actions, Docker Compose
- TOML for configuration

**Key Files:**
```
├── scenario.toml           # Evaluation configuration
├── generate_compose.py     # Generates docker-compose.yml from scenario.toml
├── record_provenance.py    # Records SHA256 digests and metadata
├── .github/workflows/
│   └── run-scenario.yml    # GitHub Actions workflow
├── results/                # Evaluation results
└── submissions/            # Historical submissions
```

**Workflow Trigger:** Push to `scenario.toml` (only on forks or non-main branches)

---

## A2A Protocol Endpoints

All agents expose these endpoints on port **9009**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/agent-card.json` | GET | Agent metadata (capabilities, skills) |
| `/v1/message:send` | POST | Send message (blocking) |
| `/v1/message:stream` | POST | Send message (SSE streaming) |
| `/v1/tasks/{id}` | GET | Get task status |
| `/v1/tasks/{id}:cancel` | POST | Cancel task |

---

## Evaluation Configuration (scenario.toml)

### Single Participant (Default)
```toml
[green_agent]
agentbeats_id = "019c066d-d16f-7bf0-9e30-4a3eac7100fc"
env = { LOG_LEVEL = "INFO" }

[[participants]]
agentbeats_id = "PARTICIPANT_AGENTBEATS_ID"
name = "participant"
env = {}

[config]
seed = 42              # For reproducibility
maxQuestions = 50      # Max questions from public.csv
timeoutSeconds = 120   # Timeout per question
participantRole = "participant"
allowNetwork = false   # Network disabled during evaluation
```

### Multi-Model Comparison (via OpenRouter)
```toml
[green_agent]
agentbeats_id = "019c066d-d16f-7bf0-9e30-4a3eac7100fc"
env = { LOG_LEVEL = "INFO" }

# GPT-4o via OpenRouter
[[participants]]
agentbeats_id = "PURPLE_AGENT_ID"
name = "gpt-4o"
env = { FINANCE_MODEL = "openrouter/openai/gpt-4o", OPENROUTER_API_KEY = "${OPENROUTER_API_KEY}" }

# Claude Sonnet via OpenRouter
[[participants]]
agentbeats_id = "PURPLE_AGENT_ID"
name = "claude-sonnet"
env = { FINANCE_MODEL = "openrouter/anthropic/claude-sonnet-4", OPENROUTER_API_KEY = "${OPENROUTER_API_KEY}" }

# Gemini Flash via OpenRouter
[[participants]]
agentbeats_id = "PURPLE_AGENT_ID"
name = "gemini-flash"
env = { FINANCE_MODEL = "openrouter/google/gemini-2.0-flash-001", OPENROUTER_API_KEY = "${OPENROUTER_API_KEY}" }

[config]
seed = 42
maxQuestions = 50
timeoutSeconds = 120
participantRole = "gpt-4o"  # Primary participant for winner determination
allowNetwork = true         # Required for OpenRouter API calls
```

---

## Scoring System

**Primary Metric:** `average_score` (0.0 - 1.0) - higher is better

**Criteria per question:**
- `correctness` - Is the answer factually correct?
- `contradiction` - Are there internal contradictions?
- `citations` - Are cited sources valid?

**Result Structure:**
```json
{
  "winner": "participant_role",
  "participants": {
    "participant": {
      "summary": {
        "total": 50,
        "passed": 42,
        "average_score": 0.84,
        "citation_valid": 38,
        "errors": 0,
        "duration_seconds": 234.5
      },
      "results": [...]
    }
  }
}
```

---

## Key Differences: Green vs Purple Agent

| Aspect | Green Agent (Evaluator) | Purple Agent (Participant) |
|--------|------------------------|----------------------------|
| **Role** | Judge/Scorer | Competitor |
| **Tools** | OFFLINE (local cache) | ONLINE (live APIs) |
| **A2A Implementation** | Custom FastAPI | a2a-sdk |
| **Network during eval** | N/A | Disabled by default (`allowNetwork=false`) |
| **LLM Provider** | model-library only | model-library + OpenRouter |

---

## Development Commands

### Purple Agent
```bash
# Install dependencies
uv sync

# Run server locally
uv run src/server.py --host 0.0.0.0 --port 9009

# Run with OpenRouter model
OPENROUTER_API_KEY=xxx FINANCE_MODEL=openrouter/openai/gpt-4o \
  uv run src/server.py --host 0.0.0.0 --port 9009

# Run tests
uv sync --extra test
uv run pytest --agent-url http://localhost:9009

# Build Docker
docker build -t purple-agent-finance .
docker run -p 9009:9009 -e SERPAPI_API_KEY=xxx -e SEC_EDGAR_API_KEY=xxx purple-agent-finance

# Run Docker with OpenRouter
docker run -p 9009:9009 \
  -e OPENROUTER_API_KEY=xxx \
  -e FINANCE_MODEL=openrouter/anthropic/claude-sonnet-4 \
  purple-agent-finance
```

### Green Agent
```bash
# Run server
uv run src/finance_green_agent/server.py --host 0.0.0.0 --port 9009

# Build Docker
docker build -t greenagentfinance .
```

### Leaderboard (Local Testing)
```bash
# Generate docker-compose from scenario
python generate_compose.py scenario.toml

# Run evaluation
docker compose up

# Record provenance
python record_provenance.py docker-compose.yml provenance.json
```

---

## Submission Flow

1. **Fork** the Leaderboard repository
2. **Edit** `scenario.toml` with your `agentbeats_id`
3. **Push** to trigger GitHub Actions
4. **Review** generated `results.json`
5. **Create PR** to submit results

---

## OpenRouter Integration

The Purple Agent supports **OpenRouter** for multi-model comparison using a single API key.

### How It Works

1. **Provider Detection:** `get_agent.py` detects the `openrouter/` prefix in `FINANCE_MODEL`
2. **OpenRouter LLM:** `openrouter_provider.py` wraps the OpenAI SDK with OpenRouter's endpoint
3. **Model Tracking:** Responses include `model_used` in the A2A artifact for comparison

### Supported Models

| Model ID | Provider |
|----------|----------|
| `openrouter/openai/gpt-4o` | OpenAI |
| `openrouter/openai/gpt-4o-mini` | OpenAI |
| `openrouter/anthropic/claude-sonnet-4` | Anthropic |
| `openrouter/anthropic/claude-opus-4` | Anthropic |
| `openrouter/google/gemini-2.0-flash-001` | Google |
| `openrouter/google/gemini-pro` | Google |
| `openrouter/meta-llama/llama-3.1-405b-instruct` | Meta |

### Usage

```bash
# Set OpenRouter API key
export OPENROUTER_API_KEY=sk-or-v1-xxx

# Use an OpenRouter model
export FINANCE_MODEL=openrouter/openai/gpt-4o

# Run the agent
uv run src/server.py --port 9009
```

### Multi-Model Evaluation Flow

```
1. scenario.toml defines N participants (same agent, different FINANCE_MODEL)
2. generate_compose.py creates N Docker services
3. docker compose up launches N instances of purple agent
4. Green Agent evaluates each participant
5. results.json contains comparison:
   {
     "winner": "claude-sonnet",
     "participants": {
       "gpt-4o": { "average_score": 0.82 },
       "claude-sonnet": { "average_score": 0.88 },
       "gemini-flash": { "average_score": 0.79 }
     }
   }
```

---

## Important Notes

- **Date hardcoded:** Both agents use April 7, 2025 as the reference date in prompts
- **Determinism:** Green agent enforces strict reproducibility (fixed seeds, pinned deps)
- **Network disabled:** During evaluation, participant agents cannot access the internet (unless using OpenRouter)
- **Citations required:** Answers must cite sources from the cache (for Green) or provide valid URLs (for Purple)
- **OpenRouter requires network:** Set `allowNetwork = true` in scenario.toml when using OpenRouter models
