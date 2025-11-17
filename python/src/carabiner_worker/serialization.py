from __future__ import annotations

import importlib
import json
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

PRIMITIVE_TYPES = (str, int, float, bool, type(None))


class EncodedKind(str, Enum):
    PRIMITIVE = "primitive"
    BASEMODEL = "basemodel"


class EncodedModelIdentifier(BaseModel):
    module: str
    name: str

    model_config = {"extra": "forbid"}


class PrimitiveEncodedValue(BaseModel):
    kind: Literal[EncodedKind.PRIMITIVE] = Field(default=EncodedKind.PRIMITIVE)
    value: Any

    model_config = {"extra": "forbid"}


class BaseModelEncodedValue(BaseModel):
    kind: Literal[EncodedKind.BASEMODEL] = Field(default=EncodedKind.BASEMODEL)
    model: EncodedModelIdentifier
    data: dict[str, Any]

    model_config = {"extra": "forbid"}


_encoded_value_adapter = TypeAdapter(
    Annotated[
        Union[PrimitiveEncodedValue, BaseModelEncodedValue],
        Field(discriminator="kind"),
    ]
)


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
        return _instantiate_serialized_model(encoded.model.module, encoded.model.name, encoded.data)
    raise ValueError(f"unsupported encoded kind: {encoded.kind!r}")


def _instantiate_serialized_model(module: str, name: str, model_data: dict[str, Any]) -> Any:
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
