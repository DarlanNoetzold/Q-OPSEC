from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import requests

from src.common.config_loader import default_config_loader
from src.common.logger import get_logger

logger = get_logger("ollama_client")


class OllamaClient:
    """Client to interact with local Ollama instance for LLM-derived features.

    Supports:
      - Connection health check
      - Risk assessment for transactions
      - Phishing detection for text
    """

    def __init__(self) -> None:
        self.config = default_config_loader.load("llm_config.yaml")
        self.base_url = self.config.get("ollama.base_url", "http://localhost:11434")
        self.model = self.config.get("ollama.model", "llama3")
        self.timeout = self.config.get("ollama.timeout", 30)
        self.gen_params = self.config.get("ollama.generation_params", {})
        self.fallbacks = self.config.get("fallbacks", {})

    def check_connection(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error("Ollama connection failed: {e}", e=e)
            return False

    def _generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Internal method to call Ollama generate API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self.gen_params
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.warning("Ollama generation error: {e}", e=e)
            return ""

    def assess_transaction_risk(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk of a transaction using LLM."""
        prompt_template = self.config.get("prompts.transaction_risk_assessment", "")
        # Simple string replacement for simulation
        prompt = prompt_template.replace("{data}", json.dumps(transaction_data, indent=2))

        response_text = self._generate(prompt)

        try:
            # Expecting JSON from LLM based on prompt instructions
            # Find JSON block if LLM added conversational text
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end != 0:
                return json.loads(response_text[start:end])
        except Exception:
            pass

        return {
            "risk_score": self.fallbacks.get("risk_score", 0.0),
            "risk_label": self.fallbacks.get("risk_label", "unknown"),
            "explanation": "LLM parsing failed or timed out"
        }

    def detect_phishing(self, text: str) -> Dict[str, Any]:
        """Detect phishing in text using LLM."""
        prompt_template = self.config.get("prompts.phishing_detection", "")
        prompt = prompt_template.replace("{text}", text)

        response_text = self._generate(prompt)

        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end != 0:
                return json.loads(response_text[start:end])
        except Exception:
            pass

        return {
            "phishing_score": self.fallbacks.get("phishing_score", 0.0),
            "phishing_label": self.fallbacks.get("phishing_label", "unknown"),
            "explanation": "LLM parsing failed or timed out"
        }