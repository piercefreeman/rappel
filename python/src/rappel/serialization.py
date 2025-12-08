"""
Serialization utilities for workflow arguments and results.

Handles conversion between Python values and protobuf WorkflowArguments.
"""

from typing import Any

# Import proto types when available
# from proto import messages_pb2 as pb


def serialize_value(value: Any) -> Any:
    """
    Serialize a Python value to protobuf WorkflowArgumentValue.

    Supports:
    - Primitives: str, int, float, bool, None
    - Collections: list, tuple, dict
    - Pydantic BaseModel
    - Exceptions
    """
    # TODO: Implement when proto is compiled
    # For now, return JSON-serializable representation
    if value is None:
        return {"type": "null", "value": None}
    elif isinstance(value, bool):
        return {"type": "bool", "value": value}
    elif isinstance(value, int):
        return {"type": "int", "value": value}
    elif isinstance(value, float):
        return {"type": "float", "value": value}
    elif isinstance(value, str):
        return {"type": "string", "value": value}
    elif isinstance(value, (list, tuple)):
        return {
            "type": "list" if isinstance(value, list) else "tuple",
            "value": [serialize_value(v) for v in value],
        }
    elif isinstance(value, dict):
        return {
            "type": "dict",
            "value": {k: serialize_value(v) for k, v in value.items()},
        }
    elif hasattr(value, "model_dump"):
        # Pydantic BaseModel
        return {
            "type": "basemodel",
            "module": type(value).__module__,
            "name": type(value).__name__,
            "data": serialize_value(value.model_dump()),
        }
    elif isinstance(value, Exception):
        import traceback

        return {
            "type": "exception",
            "exception_type": type(value).__name__,
            "module": type(value).__module__,
            "message": str(value),
            "traceback": traceback.format_exc(),
        }
    else:
        raise TypeError(f"Cannot serialize value of type {type(value)}")


def deserialize_value(data: Any) -> Any:
    """
    Deserialize a protobuf WorkflowArgumentValue to Python value.
    """
    # TODO: Implement when proto is compiled
    if not isinstance(data, dict) or "type" not in data:
        return data

    value_type = data["type"]
    value = data.get("value")

    if value_type == "null":
        return None
    elif value_type in ("bool", "int", "float", "string"):
        return value
    elif value_type == "list":
        return [deserialize_value(v) for v in value]
    elif value_type == "tuple":
        return tuple(deserialize_value(v) for v in value)
    elif value_type == "dict":
        return {k: deserialize_value(v) for k, v in value.items()}
    elif value_type == "basemodel":
        # Reconstruct Pydantic model
        import importlib

        module = importlib.import_module(data["module"])
        model_cls = getattr(module, data["name"])
        return model_cls(**deserialize_value(data["data"]))
    elif value_type == "exception":
        # Return exception info as dict (can't reconstruct arbitrary exceptions)
        return {
            "exception_type": data["exception_type"],
            "message": data["message"],
            "traceback": data["traceback"],
        }
    else:
        raise ValueError(f"Unknown value type: {value_type}")


def serialize_kwargs(kwargs: dict[str, Any]) -> Any:
    """Serialize keyword arguments for proto."""
    return {k: serialize_value(v) for k, v in kwargs.items()}


def deserialize_kwargs(data: Any) -> dict[str, Any]:
    """Deserialize keyword arguments from proto."""
    if not isinstance(data, dict):
        return {}
    return {k: deserialize_value(v) for k, v in data.items()}
