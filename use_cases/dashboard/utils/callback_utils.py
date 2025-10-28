"""Utility decorators and helpers for Dash callbacks."""

import functools
import logging
from typing import Any, Callable
from dash.exceptions import PreventUpdate


def logged_callback(callback_name: str):
    """Decorator to add logging and error handling to Dash callbacks.
    
    This decorator:
    - Logs when callback is called (with arguments)
    - Logs successful execution (with result)
    - Logs PreventUpdate (normal flow control)
    - Logs exceptions with full traceback
    
    Args:
        callback_name: Human-readable name for the callback
    
    Example:
        @app.callback(...)
        @logged_callback("delete_model")
        def delete_model(n_clicks_list):
            ...
    """
    def decorator(func: Callable) -> Callable:
        logger = logging.getLogger('dashboard.callbacks')
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Log entry with sanitized args (don't log huge data structures)
            args_str = _sanitize_args(args)
            kwargs_str = _sanitize_args(kwargs)
            logger.debug(f"{callback_name} CALLED | args={args_str} | kwargs={kwargs_str}")
            
            try:
                result = func(*args, **kwargs)
                
                # Log success
                result_str = _sanitize_result(result)
                logger.info(f"{callback_name} SUCCESS | result={result_str}")
                
                return result
                
            except PreventUpdate:
                # This is normal - callback decided not to update
                logger.debug(f"{callback_name} PREVENTED UPDATE (normal)")
                raise
                
            except Exception as e:
                # This is a real error
                logger.error(
                    f"{callback_name} FAILED | {type(e).__name__}: {e}",
                    exc_info=True
                )
                raise
                
        return wrapper
    return decorator


def _sanitize_args(obj: Any, max_length: int = 200) -> str:
    """Convert arguments to string, truncating if too long."""
    try:
        s = str(obj)
        if len(s) > max_length:
            return s[:max_length] + f"... (truncated, length={len(s)})"
        return s
    except Exception:
        return "<error converting to string>"


def _sanitize_result(result: Any, max_length: int = 500) -> str:
    """Convert result to string, truncating if too long."""
    try:
        # Handle common Dash return types
        if isinstance(result, (list, tuple)):
            if len(result) == 0:
                return str(result)
            # For lists/tuples, show first few items
            if len(result) > 3:
                sample = list(result[:3])
                return f"{sample}... ({len(result)} items total)"
        
        s = str(result)
        if len(s) > max_length:
            return s[:max_length] + f"... (truncated, length={len(s)})"
        return s
    except Exception:
        return "<error converting to string>"


def log_state_change(operation: str, before: dict, after: dict):
    """Log a state change for debugging.
    
    Args:
        operation: Description of what changed (e.g., "model_created")
        before: State before change (relevant fields only)
        after: State after change (relevant fields only)
    """
    logger = logging.getLogger('dashboard.state')
    logger.info(f"STATE CHANGE: {operation}")
    logger.debug(f"  Before: {before}")
    logger.debug(f"  After: {after}")
