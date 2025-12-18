from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel

from rappel.actions import (
    deserialize_result_payload,
    serialize_error_payload,
    serialize_result_payload,
)


class SampleModel(BaseModel):
    payload: str


@dataclass
class SampleDataclass:
    payload: str
    count: int


def test_result_round_trip_with_basemodel() -> None:
    payload = serialize_result_payload(SampleModel(payload="hello"))
    decoded = deserialize_result_payload(payload)
    assert decoded.error is None
    assert isinstance(decoded.result, SampleModel)
    assert decoded.result.payload == "hello"


def test_error_payload_serialization() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        payload = serialize_error_payload("demo.echo", exc)
    decoded = deserialize_result_payload(payload)
    assert decoded.result is None
    assert decoded.error is not None
    assert decoded.error["type"] == "RuntimeError"
    assert decoded.error["module"] == "builtins"
    assert "boom" in decoded.error["message"]
    assert "Traceback" in decoded.error["traceback"]


def test_collections_round_trip() -> None:
    payload = serialize_result_payload({"items": [1, 2, 3], "pair": (4, 5)})
    decoded = deserialize_result_payload(payload)
    assert decoded.error is None
    assert decoded.result == {"items": [1, 2, 3], "pair": (4, 5)}


def test_primitives_preserve_types() -> None:
    payload = serialize_result_payload({"count": 5, "ratio": 2.5, "flag": True, "missing": None})
    decoded = deserialize_result_payload(payload)
    assert decoded.error is None
    assert decoded.result is not None
    result = decoded.result
    assert isinstance(result["count"], int)
    assert result["count"] == 5
    assert isinstance(result["ratio"], float)
    assert result["ratio"] == 2.5
    assert result["flag"] is True
    assert result["missing"] is None


def test_result_round_trip_with_dataclass() -> None:
    """Test that dataclasses can be serialized and deserialized like Pydantic models."""
    payload = serialize_result_payload(SampleDataclass(payload="world", count=42))
    decoded = deserialize_result_payload(payload)
    assert decoded.error is None
    assert isinstance(decoded.result, SampleDataclass)
    assert decoded.result.payload == "world"
    assert decoded.result.count == 42


class ModelWithUUID(BaseModel):
    id: UUID
    name: str


class ModelWithUUIDList(BaseModel):
    ids: list[UUID]


def test_uuid_serialization() -> None:
    """Test that UUIDs are serialized as strings."""
    test_uuid = UUID("12345678-1234-5678-1234-567812345678")
    payload = serialize_result_payload({"user_id": test_uuid})
    decoded = deserialize_result_payload(payload)
    assert decoded.error is None
    # UUID is serialized as string
    assert decoded.result == {"user_id": str(test_uuid)}


def test_uuid_in_pydantic_model() -> None:
    """Test that UUIDs in Pydantic models round-trip correctly."""
    test_uuid = UUID("12345678-1234-5678-1234-567812345678")
    payload = serialize_result_payload(ModelWithUUID(id=test_uuid, name="test"))
    decoded = deserialize_result_payload(payload)
    assert decoded.error is None
    assert isinstance(decoded.result, ModelWithUUID)
    assert decoded.result.id == test_uuid
    assert decoded.result.name == "test"


def test_uuid_list_in_pydantic_model() -> None:
    """Test that lists of UUIDs in Pydantic models round-trip correctly."""
    test_uuids = [
        UUID("12345678-1234-5678-1234-567812345678"),
        UUID("87654321-4321-8765-4321-876543218765"),
    ]
    payload = serialize_result_payload(ModelWithUUIDList(ids=test_uuids))
    decoded = deserialize_result_payload(payload)
    assert decoded.error is None
    assert isinstance(decoded.result, ModelWithUUIDList)
    assert decoded.result.ids == test_uuids
