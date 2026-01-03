# config/logging_config.py
import logging
import sys
import json
from typing import Dict, Any, Optional
from datetime import datetime
import structlog
from pythonjsonlogger import jsonlogger


class StructuredLogger:
    """Structured logging configuration for Finance Judge System"""

    @staticmethod
    def setup_logging(
            log_level: str = "INFO",
            json_format: bool = False,
            log_file: Optional[str] = None
    ):
        """Setup structured logging"""

        # Configure structlog
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer() if json_format else structlog.dev.ConsoleRenderer()
            ],
            wrapper_class=structlog.BoundLogger,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Configure standard logging
        log_handlers = []

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if json_format:
            console_handler.setFormatter(jsonlogger.JsonFormatter())
        else:
            console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            console_handler.setFormatter(logging.Formatter(console_format))
        log_handlers.append(console_handler)

        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(jsonlogger.JsonFormatter())
            log_handlers.append(file_handler)

        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            handlers=log_handlers,
            force=True
        )

        # Silence noisy libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)


class FinanceLogger:
    """Finance-specific logger with context"""

    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
        self.context: Dict[str, Any] = {}

    def add_context(self, **kwargs):
        """Add context to all subsequent log messages"""
        self.context.update(kwargs)

    def clear_context(self):
        """Clear logging context"""
        self.context.clear()

    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self.logger.info(message, **{**self.context, **kwargs})

    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self.logger.warning(message, **{**self.context, **kwargs})

    def error(self, message: str, **kwargs):
        """Log error message with context"""
        self.logger.error(message, **{**self.context, **kwargs})

    def critical(self, message: str, **kwargs):
        """Log critical message with context"""
        self.logger.critical(message, **{**self.context, **kwargs})

    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self.logger.debug(message, **{**self.context, **kwargs})

    def audit(self, event: str, user: str = None, **kwargs):
        """Log audit event"""
        audit_data = {
            "event": event,
            "user": user or "system",
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        self.logger.info("audit_event", **audit_data)

    def performance(self, operation: str, duration_ms: float, **kwargs):
        """Log performance metric"""
        perf_data = {
            "operation": operation,
            "duration_ms": duration_ms,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        self.logger.info("performance_metric", **perf_data)

    def evaluation_result(self, evaluation_id: str, result: Dict[str, Any]):
        """Log evaluation result"""
        self.logger.info("evaluation_completed",
                         evaluation_id=evaluation_id,
                         result=result,
                         timestamp=datetime.utcnow().isoformat())


class LoggingMiddleware:
    """FastAPI middleware for logging"""

    def __init__(self, app):
        self.app = app
        self.logger = FinanceLogger("middleware")

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http':
            await self.log_http_request(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    async def log_http_request(self, scope, receive, send):
        """Log HTTP request/response"""
        start_time = datetime.utcnow()

        # Create logging wrapper
        async def send_wrapper(message):
            if message['type'] == 'http.response.start':
                # Log response
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds() * 1000

                self.logger.performance(
                    operation=f"http_{scope['path']}",
                    duration_ms=duration,
                    method=scope['method'],
                    path=scope['path'],
                    status_code=message['status']
                )

            await send(message)

        # Add request ID to context
        request_id = scope.get('headers', {}).get(b'x-request-id', b'').decode()
        if not request_id:
            import uuid
            request_id = str(uuid.uuid4())

        self.logger.add_context(request_id=request_id)

        # Log request
        self.logger.info("http_request",
                         method=scope['method'],
                         path=scope['path'],
                         client=scope.get('client', ('unknown', 0))[0])

        await self.app(scope, receive, send_wrapper)


# Singleton logger instances
judge_logger = FinanceLogger("judge_agent")
finance_logger = FinanceLogger("finance_agent")
a2a_logger = FinanceLogger("a2a_protocol")
mcp_logger = FinanceLogger("mcp_protocol")
sec_logger = FinanceLogger("sec_client")


def get_logger(name: str) -> FinanceLogger:
    """Get logger by name"""
    return FinanceLogger(name)