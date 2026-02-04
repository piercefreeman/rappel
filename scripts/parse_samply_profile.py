#!/usr/bin/env python3
"""Parse a samply/Firefox profiler JSON and print a simple text breakdown."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_profile(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "profile" in data and isinstance(data["profile"], dict):
        return data["profile"]
    if isinstance(data, dict):
        return data
    raise ValueError("unexpected profile format")


def schema_index(schema: Any, key: str) -> Optional[int]:
    if isinstance(schema, dict):
        value = schema.get(key)
        if isinstance(value, int):
            return value
    return None


def get_string(table: List[Any], index: Any) -> str:
    if not isinstance(index, int):
        return "<unknown>"
    if index < 0 or index >= len(table):
        return "<unknown>"
    value = table[index]
    if isinstance(value, str):
        return value
    return str(value)


def pick_thread(threads: List[Dict[str, Any]], name_filter: Optional[str]) -> Dict[str, Any]:
    if name_filter:
        for thread in threads:
            name = thread.get("name")
            if isinstance(name, str) and name_filter in name:
                return thread

    def sample_count(thread: Dict[str, Any]) -> int:
        samples = thread.get("samples", {})
        data = samples.get("data")
        if isinstance(data, list):
            return len(data)
        stacks = samples.get("stack")
        if isinstance(stacks, list):
            return len(stacks)
        return 0

    return max(threads, key=sample_count)


def iter_stack_frames(
    stack_table: Dict[str, Any],
    stack_index: int,
) -> Iterable[int]:
    data = stack_table.get("data")
    if isinstance(data, list):
        schema = stack_table.get("schema", {})
        prefix_index = schema_index(schema, "prefix")
        frame_index = schema_index(schema, "frame")
        current_stack: Optional[int] = stack_index
        while isinstance(current_stack, int):
            if current_stack < 0 or current_stack >= len(data):
                break
            row = data[current_stack]
            if isinstance(row, dict):
                frame_id = row.get("frame")
                prefix = row.get("prefix")
            else:
                frame_id = row[frame_index] if frame_index is not None else None
                prefix = row[prefix_index] if prefix_index is not None else None
            if isinstance(frame_id, int):
                yield frame_id
            if prefix is None or (isinstance(prefix, int) and prefix < 0):
                break
            current_stack = prefix if isinstance(prefix, int) else None
        return

    prefix_col = stack_table.get("prefix")
    frame_col = stack_table.get("frame")
    if not isinstance(prefix_col, list) or not isinstance(frame_col, list):
        return
    current_stack = stack_index
    while isinstance(current_stack, int):
        if current_stack < 0 or current_stack >= len(frame_col):
            break
        frame_id = frame_col[current_stack]
        if isinstance(frame_id, int):
            yield frame_id
        prefix = prefix_col[current_stack] if current_stack < len(prefix_col) else None
        if prefix is None or (isinstance(prefix, int) and prefix < 0):
            break
        current_stack = prefix if isinstance(prefix, int) else None


def build_frame_resolver(
    thread: Dict[str, Any],
    string_table: List[Any],
    symbol_map: Dict[int, str],
) -> Tuple[Dict[int, str], Dict[int, int]]:
    frame_table = thread.get("frameTable", {})
    func_table = thread.get("funcTable", {})

    func_name_cache: Dict[int, str] = {}

    def func_name(func_index: int) -> str:
        if func_index in func_name_cache:
            return func_name_cache[func_index]
        name_value: Any = None
        func_data = func_table.get("data")
        if isinstance(func_data, list):
            func_schema = func_table.get("schema", {})
            func_name_index = schema_index(func_schema, "name")
            if 0 <= func_index < len(func_data):
                entry = func_data[func_index]
                if isinstance(entry, dict):
                    name_value = entry.get("name")
                else:
                    name_value = entry[func_name_index] if func_name_index is not None else None
        else:
            name_col = func_table.get("name")
            if isinstance(name_col, list) and 0 <= func_index < len(name_col):
                name_value = name_col[func_index]
        name = get_string(string_table, name_value)
        func_name_cache[func_index] = name
        return name

    frame_to_func: Dict[int, int] = {}
    frame_to_name: Dict[int, str] = {}

    frame_data = frame_table.get("data")
    if isinstance(frame_data, list):
        frame_schema = frame_table.get("schema", {})
        frame_func_index = schema_index(frame_schema, "func")
        frame_location_index = schema_index(frame_schema, "location")
        frame_address_index = schema_index(frame_schema, "address")
        for idx, entry in enumerate(frame_data):
            if isinstance(entry, dict):
                func_index = entry.get("func")
                location = entry.get("location")
                address = entry.get("address")
            else:
                func_index = entry[frame_func_index] if frame_func_index is not None else None
                location = entry[frame_location_index] if frame_location_index is not None else None
                address = entry[frame_address_index] if frame_address_index is not None else None
            name = "<unknown>"
            if isinstance(func_index, int):
                frame_to_func[idx] = func_index
                name = func_name(func_index)
            elif location is not None:
                name = get_string(string_table, location)
            if isinstance(address, int) and address in symbol_map:
                name = symbol_map[address]
            frame_to_name[idx] = name
        return frame_to_name, frame_to_func

    frame_len = frame_table.get("length")
    if not isinstance(frame_len, int):
        func_col = frame_table.get("func")
        frame_len = len(func_col) if isinstance(func_col, list) else 0

    func_col = frame_table.get("func")
    address_col = frame_table.get("address")
    for idx in range(frame_len):
        func_index = func_col[idx] if isinstance(func_col, list) and idx < len(func_col) else None
        address = (
            address_col[idx] if isinstance(address_col, list) and idx < len(address_col) else None
        )
        name = "<unknown>"
        if isinstance(func_index, int):
            frame_to_func[idx] = func_index
            name = func_name(func_index)
        elif isinstance(address, int):
            name = f"0x{address:x}"
        if isinstance(address, int) and address in symbol_map:
            name = symbol_map[address]
        frame_to_name[idx] = name

    return frame_to_name, frame_to_func


def parse_thread_samples(
    thread: Dict[str, Any],
    string_table: List[Any],
    symbol_map: Dict[int, str],
) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
    samples = thread.get("samples", {})
    stack_table = thread.get("stackTable", {})
    frame_to_name, _ = build_frame_resolver(thread, string_table, symbol_map)

    stack_cache: Dict[int, List[str]] = {}

    def stack_names(stack_index: int) -> List[str]:
        if stack_index in stack_cache:
            return stack_cache[stack_index]
        names: List[str] = []
        for frame_index in iter_stack_frames(stack_table, stack_index):
            names.append(frame_to_name.get(frame_index, "<unknown>"))
        stack_cache[stack_index] = names
        return names

    inclusive: Dict[str, float] = defaultdict(float)
    self_time: Dict[str, float] = defaultdict(float)
    total_weight = 0.0
    total_samples = 0.0

    data = samples.get("data")
    if isinstance(data, list) and data:
        schema = samples.get("schema", {})
        stack_index_key = schema_index(schema, "stack")
        weight_index = schema_index(schema, "weight")
        if weight_index is None:
            weight_index = schema_index(schema, "duration")

        for row in data:
            if isinstance(row, dict):
                stack_index = row.get("stack")
                weight = row.get("weight") if weight_index is not None else None
            else:
                stack_index = row[stack_index_key] if stack_index_key is not None else None
                weight = row[weight_index] if weight_index is not None else None
            if not isinstance(stack_index, int):
                continue
            sample_weight = float(weight) if isinstance(weight, (int, float)) else 1.0
            total_weight += sample_weight
            total_samples += 1.0
            names = stack_names(stack_index)
            if not names:
                continue
            leaf = names[0]
            self_time[leaf] += sample_weight
            for name in names:
                inclusive[name] += sample_weight

        return inclusive, self_time, total_weight, total_samples

    stacks = samples.get("stack")
    if not isinstance(stacks, list) or not stacks:
        return {}, {}, 0.0, 0.0
    weights = samples.get("weight")

    for idx, stack_index in enumerate(stacks):
        if not isinstance(stack_index, int):
            continue
        weight = weights[idx] if isinstance(weights, list) and idx < len(weights) else None
        sample_weight = float(weight) if isinstance(weight, (int, float)) else 1.0
        total_weight += sample_weight
        total_samples += 1.0
        names = stack_names(stack_index)
        if not names:
            continue
        leaf = names[0]
        self_time[leaf] += sample_weight
        for name in names:
            inclusive[name] += sample_weight

    return inclusive, self_time, total_weight, total_samples


def format_weight(weight: float, interval_ms: Optional[float]) -> str:
    if interval_ms is None:
        if weight.is_integer():
            return f"{int(weight)} samples"
        return f"{weight:.2f} samples"
    seconds = (weight * interval_ms) / 1000.0
    if seconds < 1.0:
        return f"{seconds * 1000.0:.2f} ms"
    return f"{seconds:.2f} s"


def print_top(
    title: str, data: Dict[str, float], total: float, interval_ms: Optional[float], top: int
) -> None:
    print(title)
    if not data:
        print("  (no samples)")
        return
    items = sorted(data.items(), key=lambda item: item[1], reverse=True)
    for idx, (name, weight) in enumerate(items[:top], start=1):
        percent = (weight / total * 100.0) if total > 0 else 0.0
        weight_str = format_weight(weight, interval_ms)
        print(f"  {idx:>2}. {name}  {percent:5.1f}%  {weight_str}")


def parse_symbol_name(value: Any, string_table: Optional[List[Any]]) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, int) and isinstance(string_table, list) and 0 <= value < len(string_table):
        entry = string_table[value]
        if isinstance(entry, str):
            return entry
    return None


def parse_symbol_address(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            if text.startswith(("0x", "0X")):
                return int(text, 16)
            return int(text)
        except ValueError:
            return None
    return None


def collect_symbols(
    symbols: Any, string_table: Optional[List[Any]], output: Dict[int, str]
) -> None:
    if not isinstance(symbols, list):
        return
    for entry in symbols:
        address = None
        name = None
        if isinstance(entry, dict):
            address = parse_symbol_address(entry.get("address"))
            name = parse_symbol_name(entry.get("name"), string_table)
            if name is None:
                name = parse_symbol_name(entry.get("symbol"), string_table)
        elif isinstance(entry, (list, tuple)):
            if entry:
                address = parse_symbol_address(entry[0])
            if len(entry) > 1:
                name = parse_symbol_name(entry[-1], string_table)
        if address is not None and name:
            output[address] = name


def symbol_name_from_table(
    symbol_table: Any, symbol_index: Any, string_table: Optional[List[Any]]
) -> Optional[str]:
    if not isinstance(symbol_index, int) or not isinstance(symbol_table, list):
        return None
    if symbol_index < 0 or symbol_index >= len(symbol_table):
        return None
    entry = symbol_table[symbol_index]
    if isinstance(entry, dict):
        symbol_value = entry.get("symbol")
    elif isinstance(entry, (list, tuple)) and entry:
        symbol_value = entry[-1]
    else:
        symbol_value = None
    return parse_symbol_name(symbol_value, string_table)


def collect_sidecar_symbols(
    lib: Any, string_table: Optional[List[Any]], output: Dict[int, str]
) -> None:
    if not isinstance(lib, dict):
        return
    symbol_table = lib.get("symbol_table") or lib.get("symbolTable")
    known_addresses = lib.get("known_addresses") or lib.get("knownAddresses")
    if not isinstance(known_addresses, list):
        return
    for entry in known_addresses:
        address = None
        symbol_index = None
        name = None
        if isinstance(entry, dict):
            address = parse_symbol_address(entry.get("address"))
            symbol_index = entry.get("symbol")
            name = parse_symbol_name(entry.get("name"), string_table)
        elif isinstance(entry, (list, tuple)) and entry:
            address = parse_symbol_address(entry[0])
            if len(entry) > 1:
                symbol_index = entry[1]
        if name is None:
            name = symbol_name_from_table(symbol_table, symbol_index, string_table)
        if address is not None and name:
            output[address] = name


def load_symbol_map(path: Optional[str]) -> Dict[int, str]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    output: Dict[int, str] = {}
    string_table = None
    if isinstance(data, dict):
        string_table = data.get("stringTable")
        if string_table is None:
            string_table = data.get("string_table")

    def handle_container(container: Any) -> None:
        if not isinstance(container, dict):
            return
        collect_symbols(container.get("symbols"), string_table, output)
        symbol_table = container.get("symbolTable")
        if isinstance(symbol_table, dict):
            addresses = symbol_table.get("address")
            names = symbol_table.get("name")
            if isinstance(addresses, list) and isinstance(names, list):
                for addr, name_value in zip(addresses, names, strict=False):
                    address = parse_symbol_address(addr)
                    name = parse_symbol_name(name_value, string_table)
                    if address is not None and name:
                        output[address] = name

    if isinstance(data, dict):
        if "data" in data and "string_table" in data and isinstance(data.get("data"), list):
            for lib in data.get("data", []):
                collect_sidecar_symbols(lib, string_table, output)
        handle_container(data)
        libs = data.get("libraries") or data.get("libs")
        if isinstance(libs, list):
            for lib in libs:
                handle_container(lib)
                collect_sidecar_symbols(lib, string_table, output)
        symbolication = data.get("symbolication")
        if isinstance(symbolication, dict):
            handle_container(symbolication)
            libs = symbolication.get("libraries") or symbolication.get("libs")
            if isinstance(libs, list):
                for lib in libs:
                    handle_container(lib)
                    collect_sidecar_symbols(lib, string_table, output)
    elif isinstance(data, list):
        for lib in data:
            handle_container(lib)
            collect_sidecar_symbols(lib, string_table, output)

    return output


def resolve_syms_path(profile_path: str, override: Optional[str]) -> Optional[str]:
    if override:
        return override if os.path.exists(override) else None
    candidates = [f"{profile_path}.syms.json"]
    if profile_path.endswith(".json.gz"):
        candidates.append(profile_path[: -len(".json.gz")] + ".syms.json")
    if profile_path.endswith(".json"):
        candidates.append(profile_path[: -len(".json")] + ".syms.json")
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse samply profile JSON into a text summary.")
    parser.add_argument("profile", help="Path to samply profile JSON")
    parser.add_argument("--top", type=int, default=20, help="Number of entries to show")
    parser.add_argument("--thread", default=None, help="Substring to select a thread by name")
    parser.add_argument("--syms", default=None, help="Path to samply .syms.json sidecar")
    args = parser.parse_args()

    try:
        profile = load_profile(args.profile)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Failed to load profile: {exc}", file=sys.stderr)
        return 1

    threads = profile.get("threads", [])
    if not isinstance(threads, list) or not threads:
        print("No threads found in profile.", file=sys.stderr)
        return 1

    thread = pick_thread(threads, args.thread)
    thread_name = thread.get("name", "<unknown>")
    string_table = thread.get("stringArray")
    if not isinstance(string_table, list):
        string_table = profile.get("stringTable", [])
        if not isinstance(string_table, list):
            string_table = []
    syms_path = resolve_syms_path(args.profile, args.syms)
    symbol_map = load_symbol_map(syms_path)
    interval_ms = None
    meta = profile.get("meta", {})
    if isinstance(meta, dict):
        interval_value = meta.get("interval")
        if isinstance(interval_value, (int, float)):
            interval_ms = float(interval_value)

    inclusive, self_time, total_weight, total_samples = parse_thread_samples(
        thread, string_table, symbol_map
    )

    print("Samply profile summary")
    print(f"Thread: {thread_name}")
    print(f"Total samples: {int(total_samples)}")
    if interval_ms is not None:
        print(f"Sample interval: {interval_ms:.2f} ms")
        print(f"Approx runtime: {format_weight(total_samples, interval_ms)}")
    print(f"Total weight: {format_weight(total_weight, interval_ms)}")
    if syms_path:
        print(f"Symbols: {os.path.basename(syms_path)} ({len(symbol_map)} entries)")
    print()
    print_top("Top inclusive:", inclusive, total_weight, interval_ms, args.top)
    print()
    print_top("Top self:", self_time, total_weight, interval_ms, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
