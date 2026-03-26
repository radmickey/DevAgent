"""Error taxonomy: three exception types for the entire pipeline."""


class DevAgentError(Exception):
    """Base error for all DevAgent exceptions."""


class TransientError(DevAgentError):
    """Temporary failure — safe to retry (timeouts, rate limits, network)."""


class PermanentError(DevAgentError):
    """Unrecoverable failure — do not retry (missing resource, auth, bad input)."""


class DegradedError(DevAgentError):
    """Service unavailable but pipeline can continue with reduced context."""
