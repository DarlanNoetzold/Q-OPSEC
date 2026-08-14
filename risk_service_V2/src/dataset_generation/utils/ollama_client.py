from __future__ import annotations

import json
from typing import Any, Dict

import requests

from src.common.config_loader import default_config_loader
from src.common.logger import get_logger

logger = get_logger("ollama_client")


class OllamaClient:
    """Client for interacting with local Ollama LLM instance."""

    def __init__(self) -> None:
        """Initialize Ollama client with config."""
        llm_config = default_config_loader.load("llm_config.yaml")

        self.base_url = llm_config.get("ollama", {}).get("base_url", "http://localhost:11434")
        self.model = llm_config.get("ollama", {}).get("model", "llama3:8b")
        self.timeout = llm_config.get("ollama", {}).get("timeout", 30)

        self.generation_params = llm_config.get("generation", {})
        self.prompts = llm_config.get("prompts", {})
        self.fallback_values = llm_config.get("fallback_values", {})

        logger.info(f"Ollama client initialized: {self.base_url}, model={self.model}")

    def test_connection(self) -> bool:
        """Test if Ollama server is reachable and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)

            if response.status_code != 200:
                logger.warning(f"Ollama server returned status {response.status_code}")
                return False

            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]  # Remove :tag

            # Check if model exists (with or without tag)
            model_base = self.model.split(":")[0]
            if model_base not in model_names and self.model not in [m.get("name", "") for m in models]:
                logger.warning(f"Model '{self.model}' not found. Available: {[m.get('name') for m in models]}")
                return False

            logger.info(f"✅ Ollama connection OK. Model '{self.model}' available.")
            return True

        except requests.exceptions.ConnectionError:
            logger.warning(f"❌ Cannot connect to Ollama at {self.base_url}")
            return False
        except Exception as e:
            logger.error(f"Ollama test failed: {e}")
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama."""
        try:
            params = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.generation_params.get("temperature", 0.7)),
                    "top_p": kwargs.get("top_p", self.generation_params.get("top_p", 0.9)),
                    "num_predict": kwargs.get("max_tokens", self.generation_params.get("max_tokens", 256)),
                }
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=params,
                timeout=self.timeout
            )

            if response.status_code != 200:
                logger.warning(f"Generation failed: {response.status_code}")
                return ""

            return response.json().get("response", "").strip()

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""

    def assess_transaction_risk(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess transaction risk using LLM."""
        try:
            prompt_template = self.prompts.get("risk_assessment", {}).get("template", "")

            if not prompt_template:
                return self.fallback_values.get("risk_assessment", {"risk_score": 0.5, "reasoning": "No template"})

            # Safe format - replace missing keys with "N/A"
            safe_context = {
                "event_type": context.get("event_type", "N/A"),
                "amount": context.get("amount", "N/A"),
                "channel": context.get("channel", "N/A"),
                "user_risk_class": context.get("user_risk_class", "N/A"),
                "message_text": context.get("message_text", "N/A")
            }

            prompt = prompt_template.format(**safe_context)
            response = self.generate(prompt, temperature=0.3)

            if not response:
                return self.fallback_values.get("risk_assessment", {"risk_score": 0.5, "reasoning": "No response"})

            try:
                result = json.loads(response)
                return {
                    "risk_score": float(result.get("risk_score", 0.5)),
                    "reasoning": result.get("reasoning", "")
                }
            except json.JSONDecodeError:
                score = 0.5
                if "high risk" in response.lower():
                    score = 0.8
                elif "low risk" in response.lower():
                    score = 0.2

                return {"risk_score": score, "reasoning": response[:200]}

        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return self.fallback_values.get("risk_assessment", {"risk_score": 0.5, "reasoning": str(e)})

    def detect_phishing(self, message_text: str) -> Dict[str, Any]:
        """Detect phishing in message text."""
        try:
            prompt_template = self.prompts.get("phishing_detection", {}).get("template", "")

            if not prompt_template:
                return self.fallback_values.get("phishing_detection", {
                    "is_phishing": False,
                    "sentiment_score": 0.0,
                    "urgency_score": 0.0
                })

            prompt = prompt_template.format(message_text=message_text)
            response = self.generate(prompt, temperature=0.2)

            if not response:
                return self.fallback_values.get("phishing_detection", {
                    "is_phishing": False,
                    "sentiment_score": 0.0,
                    "urgency_score": 0.0
                })

            try:
                result = json.loads(response)
                return {
                    "is_phishing": bool(result.get("is_phishing", False)),
                    "sentiment_score": float(result.get("sentiment_score", 0.0)),
                    "urgency_score": float(result.get("urgency_score", 0.0))
                }
            except json.JSONDecodeError:
                is_phishing = any(word in message_text.lower() for word in [
                    "urgent", "verify", "suspended", "click here", "confirm", "password"
                ])

                return {
                    "is_phishing": is_phishing,
                    "sentiment_score": 0.5,
                    "urgency_score": 0.7 if is_phishing else 0.3
                }

        except Exception as e:
            logger.error(f"Phishing detection failed: {e}")
            return self.fallback_values.get("phishing_detection", {
                "is_phishing": False,
                "sentiment_score": 0.0,
                "urgency_score": 0.0
            })