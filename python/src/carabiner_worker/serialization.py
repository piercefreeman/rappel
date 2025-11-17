from __future__ import annotations

import importlib
import json
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Union

from pydantic import BaseModel, Field, TypeAdapter

PRIMITIVE_TYPES = (str, int, float, bool, type(None))


@dataclass
class ActionCall:
    module: str
    action: str
    kwargs: dict[str, Any]


@dataclass
class ActionResultPayload:
    result: Any | None
    error: dict[str, str] | None


class EncodedKind(str, Enum):
    PRIMITIVE = "primitive"
    BASEMODEL = "basemodel"


class EncodedModelIdentifier(BaseModel):
    module: str
    name: str

    model_config = {"extra": "forbid"}


class PrimitiveEncodedValue(BaseModel):
    kind: EncodedKind = Field(default=EncodedKind.PRIMITIVE, const=True)
    value: Any

    model_config = {"extra": "forbid"}


class BaseModelEncodedValue(BaseModel):
    kind: EncodedKind = Field(default=EncodedKind.BASEMODEL, const=True)
    model: EncodedModelIdentifier
    data: dict[str, Any]

    model_config = {"extra": "forbid"}


_encoded_value_adapter = TypeAdapter(
    Annotated[
        Union[PrimitiveEncodedValue, BaseModelEncodedValue],
        Field(discriminator="kind"),
    ]
)


def serialize_action_call(module: str, action: str, /, **kwargs: Any) -> bytes:
    """Serialize an action name and keyword arguments into bytes."""
    if not isinstance(module, str) or not module:
        raise ValueError("action module must be a non-empty string")
    encoded_kwargs = {key: _encode_value(value) for key, value in kwargs.items()}
    payload = {"module": module, "action": action, "kwargs": encoded_kwargs}
    return _dumps(payload)


def deserialize_action_call(payload: bytes) -> ActionCall:
    """Deserialize a payload into an action invocation."""
    data = _loads(payload)
    module = data.get("module")
    if not isinstance(module, str) or not module:
        raise ValueError("payload missing module name")
    action = data.get("action")
    if not isinstance(action, str) or not action:
        raise ValueError("payload missing action name")
    kwargs_data = data.get("kwargs", {})
    if not isinstance(kwargs_data, dict):
        raise ValueError("payload kwargs must be an object")
    kwargs = {key: _decode_value(value) for key, value in kwargs_data.items()}
    return ActionCall(module=module, action=action, kwargs=kwargs)


def serialize_result_payload(value: Any) -> bytes:
    """Serialize a successful action result."""
    return _dumps({"result": _encode_value(value)})


def serialize_error_payload(action: str, exc: BaseException) -> bytes:
    """Serialize an error raised during action execution."""
    error_payload = {
        "error": {
            "action": action,
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    }
    return _dumps(error_payload)


def deserialize_result_payload(payload: bytes) -> ActionResultPayload:
    """Deserialize bytes produced by serialize_result_payload/error."""
    data = _loads(payload)
    if "error" in data:
        error = data["error"]
        if not isinstance(error, dict):
            raise ValueError("error payload must be an object")
        return ActionResultPayload(result=None, error=error)
    if "result" not in data:
        raise ValueError("result payload missing 'result' field")
    return ActionResultPayload(result=_decode_value(data["result"]), error=None)


def _encode_value(value: Any) -> dict[str, Any]:
    if isinstance(value, PRIMITIVE_TYPES):
        return {"kind": "primitive", "value": value}
    if _is_base_model(value):
        model_class = value.__class__
        if hasattr(value, "model_dump"):
            model_data = value.model_dump(mode="python")  # type: ignore[attr-defined]
        elif hasattr(value, "dict"):
            model_data = value.dict()  # type: ignore[attr-defined]
        else:  # pragma: no cover - fallback path
            model_data = value.__dict__
        return {
            "kind": "basemodel",
            "model": {
                "module": model_class.__module__,
                "name": model_class.__qualname__,
            },
            "data": model_data,
        }
    raise TypeError(f"unsupported value type {type(value)!r}")


def _decode_value(data: Any) -> Any:
    if not isinstance(data, dict):
        raise ValueError("encoded values must be objects")
    encoded = _encoded_value_adapter.validate_python(data)
    if encoded.kind is EncodedKind.PRIMITIVE:
        return encoded.value
    if encoded.kind is EncodedKind.BASEMODEL:
        return _instantiate_serialized_model(
            encoded.model.module, encoded.model.name, encoded.data
        )
    raise ValueError(f"unsupported encoded kind: {encoded.kind!r}")


def _instantiate_serialized_model(
    module: str, name: str, model_data: dict[str, Any]
) -> Any:
    cls = _import_symbol(module, name)
    if hasattr(cls, "model_validate"):
        return cls.model_validate(model_data)  # type: ignore[attr-defined]
    return cls(**model_data)


def _is_base_model(value: Any) -> bool:
    return isinstance(value, BaseModel)


def _import_symbol(module: str, qualname: str) -> Any:
    module_obj = importlib.import_module(module)
    attr: Any = module_obj
    for part in qualname.split("."):
        attr = getattr(attr, part)
    if not isinstance(attr, type):
        raise ValueError(f"{qualname} from {module} is not a class")
    return attr


def _dumps(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _loads(payload: bytes) -> Any:
    if isinstance(payload, bytes):
        text = payload.decode("utf-8")
    elif isinstance(payload, str):  # pragma: no cover - convenience path
        text = payload
    else:
        raise TypeError("payload must be bytes or str")
    return json.loads(text)
