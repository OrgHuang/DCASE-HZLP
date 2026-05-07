#!/usr/bin/env python
"""HTTP reward server for AudioMCQ statistical PPO reward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiohttp import web

from audio_mcq_statistical_reward import load_reference_index, score_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("/home/org/DCASE/AudioMCQ-StrongAC-GeminiCoT-complete/data_acoustic_cot_no_teacher.jsonl"),
        help="Source jsonl used to build the reference prompt and Gemini CoT index.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the reward server.")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind the reward server.")
    return parser.parse_args()


async def handle_health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def handle_reward(request: web.Request) -> web.Response:
    app = request.app
    payload = await request.json()
    messages = payload.get("messages", [])
    scores = []
    debug = []
    for message in messages:
        score, breakdown = score_message(str(message), app["reference_index"])
        scores.append(float(score))
        debug.append(breakdown)

    return web.json_response({"scores": scores, "debug": debug})


def main() -> None:
    args = parse_args()
    reference_index = load_reference_index(args.input_jsonl)
    app = web.Application()
    app["reference_index"] = reference_index
    app.add_routes(
        [
            web.get("/health", handle_health),
            web.post("/reward", handle_reward),
        ]
    )
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
