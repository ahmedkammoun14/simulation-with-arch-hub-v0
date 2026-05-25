"""Intent Engine that uses Ollama qwen2.5 to transform intentions into SLOs.

This module exposes IntentEngine which calls a local Ollama server via the
REST /api/chat endpoint and parses a strict JSON array response into SLO models.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import List

import httpx

from models.schemas import SLO, SLOResponse


logger = logging.getLogger(__name__)


class IntentEngine:
    """Local Intent Engine that converts natural language intentions to SLOs."""

    def __init__(self, model: str = "qwen2.5", ollama_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")

    def _build_system_prompt(self) -> str:
        """Build a strict system prompt instructing the model to output JSON only.

        The model must:
        - Return ONLY a JSON array (no surrounding text or markdown).
        - Extract numeric threshold values directly from the user's intention text when present.
        - If a metric is mentioned without an explicit numeric value, substitute a reasonable default:
            latency -> 50 (ms), cpu_usage -> 75 (%), ram_usage -> 80 (%)
        - Only include SLO objects for metrics explicitly mentioned in the user's intention.
        - Preserve the exact output schema: metric, operator, threshold, unit.
        - Allowed metrics: "latency", "cpu_usage", "ram_usage".
        - Allowed operators: "<", "<=", ">", ">=". Units: "ms" for latency, "%" for cpu/ram.

        Example output:
        [{"metric":"latency","operator":"<","threshold":20,"unit":"ms"},
         {"metric":"cpu_usage","operator":"<","threshold":70,"unit":"%"}]
        """

        return (
            "You are an assistant that MUST respond with ONLY a JSON array and nothing else. "
            "Do not include explanations, markdown, or any text outside the JSON array. "
            "Extract numeric threshold values directly from the user's intention text when present. "
            "If a metric is mentioned without an explicit value, use these defaults: threshold=50 unit=ms for latency, threshold=75 unit=% for cpu_usage, threshold=80 unit=% for ram_usage. threshold must always be a number, never a string. "
            "Only include SLO objects for metrics explicitly mentioned in the intention. "
            "Each item must be an object with keys: metric, operator, threshold, unit. "
            "Allowed metrics: \"latency\", \"cpu_usage\", \"ram_usage\". "
            "Allowed operators: \"<\", \"<=\", \">\", \">=\". "
            "Unit must be \"ms\" for latency and \"%\" for cpu/ram. "
            "Example output: "
            "[{\"metric\":\"latency\",\"operator\":\"<\",\"threshold\":20,\"unit\":\"ms\"},"
            "{\"metric\":\"cpu_usage\",\"operator\":\"<\",\"threshold\":70,\"unit\":\"%\"}]"
        )

    def _extract_json_array(self, text: str) -> str:
        """Extract the first JSON array found in the text.

        This is defensive: the model should already return only JSON, but some
        responses include backticks or surrounding text; extract between first
        '[' and last ']' to recover the array.
        """

        if not text:
            raise ValueError("Empty response text")
        # Strip common code fences
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            # remove triple backticks and optional language
            parts = text.split("\n", 1)
            if len(parts) > 1:
                text = parts[1].rsplit("\n", 1)[0]
            else:
                text = text.strip("`\n ")

        first = text.find("[")
        last = text.rfind("]")
        if first == -1 or last == -1 or last < first:
            raise ValueError("No JSON array found in response")
        return text[first : last + 1]

    async def transform(self, intent_id: str, intention: str) -> SLOResponse:
        """Transform a natural-language intention into structured SLOs.

        Calls the local Ollama /api/chat endpoint and parses the model output.
        On error returns an SLOResponse with an empty slos list.
        """

        system_prompt = self._build_system_prompt()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": intention},
            ],
            "stream": False,
        }

        async with httpx.AsyncClient(base_url=self.ollama_url, timeout=60.0) as client:
            try:
                resp = await client.post("/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()

                # Ollama response format: {"message": {"role": "assistant", "content": "..."}, "done": true}
                assistant_text = data.get("message", {}).get("content", "")
                if not assistant_text:
                    raise ValueError("Empty content from Ollama response")

                # Extract JSON array and parse
                json_array_text = self._extract_json_array(assistant_text)
                parsed = json.loads(json_array_text)
                slos: List[SLO] = [SLO(**item) for item in parsed]
                return SLOResponse(intent_id=intent_id, slos=slos)

            except Exception as exc:  # network, parsing, validation
                logger.exception("IntentEngine.transform failed for intent %s", intent_id)
                return SLOResponse(intent_id=intent_id, slos=[])


async def main() -> None:
    """Quick local test harness for the IntentEngine."""

    engine = IntentEngine()
    intent_id = f"test-{datetime.now(timezone.utc).isoformat()}"
    intention = "Keep latency below 30ms and CPU usage below 70% for the service database"
    result = await engine.transform(intent_id, intention)
    logger.info("SLOResponse: %s", result.json())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

