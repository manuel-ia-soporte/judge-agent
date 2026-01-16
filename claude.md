# Claude Code Memory - Financial Intelligence Governance Platform

## Project Overview

This is a **multi-agent financial intelligence system** designed for AI governance in financial analysis. The system analyzes SEC filings and evaluates analysis quality through specialized agents.

**Main Purpose**: Perform trustworthy, auditable AI-generated financial intelligence at scale.

**Target Users**: Financial analysts, compliance officers, and AI governance teams.

---

## Architecture

### Architectural Style

- **Hexagonal Architecture (Ports & Adapters)**: Core domain isolated from external concerns
- **Domain-Driven Design (DDD)**: Aggregates, value objects, domain services
- **CQRS**: Commands and queries separated
- **Async-first**: Full async implementation with FastAPI

### Layer Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                          │
│                    main.py - Port 8000                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      AGENTS LAYER                               │
│  ┌────────────────────┐        ┌────────────────────┐          │
│  │   FinanceAgent     │        │   JudgeAgent       │          │
│  │   (Analysis)       │───────►│   (Evaluation)     │          │
│  └────────────────────┘        └────────────────────┘          │
│              │                          │                       │
│              └───────── AgentRegistry ──┘                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   APPLICATION LAYER                             │
│  Use Cases: AnalyzeCompany, AssessRisk, EvaluateAnalysis       │
│  Commands: AnalyzeCompanyCommand, CompareCompaniesCommand       │
│  Ports: Interfaces defining contracts                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    DOMAIN LAYER (Core)                          │
│  Services: Financial, Operational, Strategic, Risk, Rubrics     │
│  Models: SECDocument, FinancialAnalysis, Agent, RubricScore     │
│  Entities: Enums, Events, Value Objects                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                 INFRASTRUCTURE LAYER                            │
│  Adapters: SEC Edgar, LLM Client, Financial Data                │
│  MCP/A2A: Inter-agent communication                             │
│  External: External APIs (SEC, OpenAI, Anthropic, Yahoo)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
agent-judge-deepseek/
├── main.py                          # FastAPI entry point + DependencyContainer
├── requirements.txt                 # Python dependencies
├── Dockerfile & docker-compose.yml  # Container orchestration
├── prd.json                         # Product requirements document
├── agent.md                         # Agent specification
│
├── domain/                          # Core business logic layer
│   ├── models/                      # Domain entities and value objects
│   │   ├── finance.py              # SECDocument, FinancialAnalysis
│   │   ├── agent.py                # AgentCapabilities, AgentMetrics
│   │   ├── evaluation.py           # RubricCategory, RubricScore
│   │   ├── entities.py             # Base entities
│   │   ├── enums.py                # AnalysisType, RiskCategory, etc.
│   │   └── events.py               # Domain events
│   │
│   ├── services/                    # Domain services (pure business logic)
│   │   ├── financial_analysis_service.py
│   │   ├── operational_analysis_service.py
│   │   ├── strategic_analysis_service.py
│   │   ├── risk_assessment_service.py
│   │   ├── rubrics_service.py
│   │   └── evaluation_service.py
│   │
│   ├── repositories/                # Data persistence interfaces
│   └── entities/                    # DDD entities
│
├── application/                     # Application layer
│   ├── use_cases/                   # Business process coordination
│   │   ├── analyze_company_use_case.py
│   │   ├── assess_risk_use_case.py
│   │   ├── evaluate_analysis.py
│   │   ├── score_rubrics.py
│   │   └── compare_companies_use_case.py
│   │
│   ├── ports/                       # Port interfaces
│   │   ├── analysis_ports.py        # FinancialAnalysisPort, etc.
│   │   └── sec_filing_port.py
│   │
│   ├── dtos/                        # Data Transfer Objects
│   │   └── analysis_dtos.py
│   │
│   ├── commands/                    # Command pattern (CQRS)
│   │   ├── analyze_company_command.py
│   │   └── compare_companies_command.py
│   │
│   ├── queries/                     # Query pattern (CQRS)
│   ├── policies/                    # Business policies
│   └── interfaces/                  # Application interfaces
│
├── infrastructure/                  # External integrations
│   ├── adapters/                    # Port implementations
│   │   ├── analysis_adapters.py     # Financial, Operational, Strategic adapters
│   │   ├── sec_edgar_adapter.py     # SEC EDGAR API integration
│   │   ├── financial_data_adapter.py
│   │   └── evaluation_adapter.py
│   │
│   ├── llm/                         # LLM integration
│   │   └── llm_client.py           # Multi-provider (OpenAI, Anthropic)
│   │
│   ├── mcp/                         # Model Context Protocol servers
│   │   ├── judge_mcp_server.py
│   │   ├── finance_mcp_server.py
│   │   └── mcp_client.py
│   │
│   ├── a2a/                         # Agent-to-Agent communication
│   │   ├── a2a_server.py
│   │   ├── a2a_client.py
│   │   ├── agent_router.py
│   │   ├── a2a_orchestrator.py
│   │   └── message_broker.py
│   │
│   ├── external/                    # External service adapters
│   ├── metrics/                     # Monitoring & observability
│   ├── persistence/                 # Data storage layer
│   ├── audit/                       # Audit trail logging
│   └── sec_edgar/                   # SEC EDGAR specific integration
│
├── agents/                          # Multi-agent orchestration
│   ├── finance_agent/               # Financial analysis agent
│   │   ├── core/
│   │   │   ├── finance_agent.py     # Main agent orchestrator
│   │   │   └── agent_capabilities.py
│   │   │
│   │   ├── strategies/
│   │   │   ├── analysis_strategy.py  # FullAnalysisStrategy
│   │   │   └── comparison_strategy.py
│   │   │
│   │   ├── analyzers/                # Analysis implementations
│   │   │   ├── financial_analyzer.py
│   │   │   ├── operational_analyzer.py
│   │   │   ├── strategic_analyzer.py
│   │   │   ├── risk_analyzer.py
│   │   │   ├── llm_risk_analyzer.py
│   │   │   └── hybrid_risk_analyzer.py
│   │   │
│   │   ├── factories/
│   │   │   └── analyzer_factory.py
│   │   │
│   │   └── sec_analyzer.py
│   │
│   ├── judge_agent/                 # Evaluation agent
│   │   ├── judge_agent.py           # Judge orchestrator
│   │   └── rubrics_evaluator.py
│   │
│   ├── registry/
│   │   └── agent_registry.py
│   │
│   └── shared/                      # Shared agent utilities
│
├── contracts/                       # API contracts
│   ├── api/
│   │   ├── requests/
│   │   │   └── analysis_requests.py
│   │   └── responses/
│   │       └── analysis_responses.py
│   │
│   ├── evaluation_contracts.py
│   ├── finance_contracts.py
│   ├── judge_contracts.py
│   ├── benchmark_contracts.py
│   ├── events/
│   └── integration/
│
├── config/                          # Configuration
│   ├── settings.py                  # Pydantic settings
│   └── logging_config.py
│
├── tests/                           # Test suite
│   ├── test_judge_agent.py
│   ├── test_evaluation.py
│   ├── test_contracts.py
│   └── test_a2a.py
│
├── benchmark/                       # Benchmarking utilities
├── specs/                           # Specifications
└── .well-known/                     # Web standards
```

---

## Key Components

### Agents

| Agent | Location | Responsibility |
|-------|----------|----------------|
| **FinanceAgent** | `agents/finance_agent/core/finance_agent.py` | Analyzes companies using SEC documents (10-K, 10-Q). Executes financial, operational, strategic, and risk analysis |
| **JudgeAgent** | `agents/judge_agent/judge_agent.py` | Evaluates analysis quality using predefined rubrics (factual accuracy, source fidelity, etc.) |

### Domain Services

| Service | Location | Responsibility |
|---------|----------|----------------|
| **FinancialAnalysisService** | `domain/services/financial_analysis_service.py` | Extract financial metrics from SEC documents |
| **OperationalAnalysisService** | `domain/services/operational_analysis_service.py` | Assess operational efficiency |
| **StrategicAnalysisService** | `domain/services/strategic_analysis_service.py` | Evaluate strategic position |
| **RiskAssessmentService** | `domain/services/risk_assessment_service.py` | Categorize and score risks |
| **RubricsService** | `domain/services/rubrics_service.py` | Manage evaluation rubrics |

### Use Cases

| Use Case | Location | Responsibility |
|----------|----------|----------------|
| **AnalyzeCompanyUseCase** | `application/use_cases/analyze_company_use_case.py` | Orchestrate multi-dimensional company analysis |
| **AssessRiskUseCase** | `application/use_cases/assess_risk_use_case.py` | Coordinate risk assessment |
| **EvaluateAnalysisUseCase** | `application/use_cases/evaluate_analysis.py` | Orchestrate analysis evaluation |
| **ScoreRubricsUseCase** | `application/use_cases/score_rubrics.py` | Apply rubric scoring |

### Infrastructure Adapters

| Adapter | Location | Responsibility |
|---------|----------|----------------|
| **SECEdgarAdapter** | `infrastructure/adapters/sec_edgar_adapter.py` | Load SEC filings |
| **LLMClient** | `infrastructure/llm/llm_client.py` | Multi-provider LLM calls (OpenAI, Anthropic) |
| **FinancialAnalysisAdapter** | `infrastructure/adapters/analysis_adapters.py` | Bridge domain services to external systems |

---

## Design Patterns

### Architectural Patterns
- **Hexagonal Architecture**: Ports define contracts, adapters implement them
- **Dependency Injection**: `DependencyContainer` in `main.py` manages all dependencies
- **CQRS**: Commands in `application/commands/`, queries in `application/queries/`

### Behavioral Patterns
- **Strategy Pattern**: `AnalysisStrategy` → `FullAnalysisStrategy`
- **Factory Pattern**: `AnalyzerFactory`, `AgentRegistry`
- **Adapter Pattern**: `SECEdgarAdapter`, `LLMClient`

### DDD Patterns
- **Aggregate Roots**: `FinancialAnalysis`, `Agent`
- **Value Objects**: `AgentCapabilities`, `RubricScore`, `AgentMetrics`
- **Domain Services**: Pure business logic (no I/O)
- **Domain Events**: State change notifications

---

## External Dependencies

### Main Dependencies
| Package | Purpose |
|---------|---------|
| FastAPI 0.115.0+ | Web framework |
| Pydantic 2.10.0+ | Data validation |
| httpx 0.25.0+ | Async HTTP client |
| sec-edgar-downloader 5.0.3 | SEC filing downloads |
| edgartools 5.6.4 | SEC EDGAR tools |
| yfinance 0.2.33 | Stock market data |
| mcp 1.2.0, fastmcp 2.1.0 | Model Context Protocol |
| python-a2a[server,mcp] | Agent-to-Agent communication |
| redis 5.0.0 | Caching & message broker |

### External APIs
- **SEC EDGAR API**: Retrieve SEC filings (10-K, 10-Q, 8-K)
- **LLM APIs**: OpenAI (GPT-4), Anthropic (Claude 3)
- **Yahoo Finance**: Market data via yfinance

---

## Configuration

### Settings Location
`config/settings.py` - Pydantic BaseSettings

### Key Environment Variables
```
OPENAI_API_KEY or ANTHROPIC_API_KEY  # LLM provider
SEC_API_KEY                           # SEC API access
LOG_LEVEL                             # Logging level (INFO default)
MCP_HOST / MCP_PORT                   # MCP server config (0.0.0.0:8000)
A2A_HOST / A2A_PORT                   # A2A server config (0.0.0.0:8001)
```

### Default Settings
- `SEC_RATE_LIMIT`: 10 requests/sec
- `CACHE_TTL`: 3600 seconds (1 hour)
- `MAX_CONCURRENT_EVALUATIONS`: 5
- `EVALUATION_TIMEOUT`: 30 seconds
- `PASS_THRESHOLD`: 1.5

---

## Request Flow Example

### Company Analysis Flow
```
POST /api/v1/analysis/analyze
    │
    ▼
analysis_router (validates AnalyzeCompanyRequest)
    │
    ▼
FinanceAgent.analyze(AnalyzeCompanyCommand)
    │
    ▼
FullAnalysisStrategy.execute()
    ├── SECEdgarAdapter.find_by_cik() → [SECDocument...]
    ├── FinancialAnalyzer.analyze()
    ├── OperationalAnalyzer.analyze()
    ├── StrategicAnalyzer.analyze()
    └── HybridRiskAnalyzer.analyze()
    │
    ▼
AnalysisResultDTO → JSON Response
```

### Evaluation Flow
```
JudgeAgent.evaluate(analysis_result)
    │
    ▼
EvaluateAnalysisUseCase.execute()
    │
    ▼
RubricsService.score() → RubricScore[]
    │
    ▼
EvaluationResult (pass/fail + scores)
```

---

## Development Guidelines

### Adding a New Analyzer
1. Create analyzer in `agents/finance_agent/analyzers/`
2. Implement domain service in `domain/services/`
3. Define port interface in `application/ports/`
4. Create adapter in `infrastructure/adapters/`
5. Register in `AnalyzerFactory`
6. Add to strategy execution

### Adding a New Rubric
1. Define rubric in `domain/models/evaluation.py`
2. Add scoring logic in `domain/services/rubrics_service.py`
3. Update `DEFAULT_RUBRICS` in `config/settings.py`

### Adding a New Use Case
1. Create use case in `application/use_cases/`
2. Define command in `application/commands/`
3. Register in `DependencyContainer` (main.py)
4. Add route in appropriate router

### Testing
- Tests are located in `tests/`
- Use `pytest` and `pytest-asyncio`
- Mock external dependencies (LLM, SEC API)

---

## Running the Project

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker-compose up --build
```

### Health Check
```
GET /health
GET /api/v1/agents  # List registered agents
```

---

## Key Files to Know

| File | Importance |
|------|------------|
| `main.py` | Application entry point, dependency injection setup |
| `agents/finance_agent/core/finance_agent.py` | Main finance agent logic |
| `agents/judge_agent/judge_agent.py` | Evaluation agent logic |
| `domain/services/risk_assessment_service.py` | Risk scoring logic |
| `infrastructure/adapters/sec_edgar_adapter.py` | SEC data retrieval |
| `infrastructure/llm/llm_client.py` | LLM integration |
| `config/settings.py` | All configuration settings |
| `contracts/` | API request/response schemas |

---

## Common Tasks

### Modify Risk Assessment Logic
Edit: `domain/services/risk_assessment_service.py`

### Change LLM Provider/Model
Edit: `infrastructure/llm/llm_client.py`

### Add New API Endpoint
1. Add request/response in `contracts/api/`
2. Create route in appropriate router
3. Wire up use case in `main.py`

### Modify Evaluation Rubrics
Edit: `domain/services/rubrics_service.py` and `domain/models/evaluation.py`
