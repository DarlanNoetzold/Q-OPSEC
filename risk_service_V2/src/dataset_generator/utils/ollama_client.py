import json
import requests
from typing import Dict, Any, Optional
from src.common.logger import log


class OllamaClient:

    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama client.

        Args:
            config: LLM configuration dictionary
        """
        self.base_url = config['ollama']['base_url']
        self.model = config['ollama']['model']
        self.timeout = config['ollama']['timeout']
        self.generation_params = config['ollama']['generation_params']
        self.prompts = config['prompts']
        self.fallback = config['fallback']

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test connection to Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            log.info(f"✓ Connected to Ollama at {self.base_url}")
        except Exception as e:
            log.warning(f"⚠ Could not connect to Ollama: {e}")
            log.warning("LLM features will use fallback values")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Ollama.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)

        Returns:
            Generated text
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.generation_params['temperature'],
                "top_p": self.generation_params['top_p'],
                "num_predict": self.generation_params['max_tokens']
            }
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

            result = response.json()
            return result.get('response', '')

        except Exception as e:
            log.error(f"Ollama generation failed: {e}")
            return ""

    def assess_risk(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess transaction risk using LLM.

        Args:
            transaction_data: Dictionary with transaction features

        Returns:
            Dictionary with risk assessment
        """
        # Build prompt from template
        system_prompt = self.prompts['risk_scoring']['system']
        user_prompt = self.prompts['risk_scoring']['user_template'].format(
            amount=transaction_data.get('amount', 0),
            currency=transaction_data.get('currency', 'USD'),
            transaction_type=transaction_data.get('transaction_type', 'unknown'),
            channel=transaction_data.get('channel', 'unknown'),
            account_age_days=transaction_data.get('account_age_days', 0),
            is_new_recipient=transaction_data.get('is_new_recipient', False),
            country_change=transaction_data.get('country_change_since_last_session', False),
            device_change=transaction_data.get('is_new_device', False),
            mfa_used=transaction_data.get('mfa_used', False),
            hour_of_day=transaction_data.get('hour_of_day', 12),
            day_of_week=transaction_data.get('day_of_week', 'Monday'),
            message_text=transaction_data.get('message_text', 'N/A')
        )

        # Generate response
        response_text = self.generate(user_prompt, system_prompt)

        # Parse JSON response
        try:
            # Extract JSON from response (in case there's extra text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)

                # Add metadata
                result['llm_model_name'] = self.model
                result['llm_model_version'] = 'ollama_local'
                result['llm_prompt_version'] = 'v1'

                return result
            else:
                raise ValueError("No JSON found in response")

        except Exception as e:
            log.warning(f"Failed to parse LLM response: {e}")
            log.debug(f"Raw response: {response_text[:200]}")
            return self._get_fallback_risk_assessment()

    def detect_phishing(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect phishing in message text.

        Args:
            message_data: Dictionary with message and context

        Returns:
            Dictionary with phishing detection results
        """
        system_prompt = self.prompts['phishing_detection']['system']
        user_prompt = self.prompts['phishing_detection']['user_template'].format(
            message_text=message_data.get('message_text', ''),
            amount=message_data.get('amount', 0),
            is_new_recipient=message_data.get('is_new_recipient', False),
            sender_reputation=message_data.get('sender_reputation', 'unknown')
        )

        response_text = self.generate(user_prompt, system_prompt)

        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found")

        except Exception as e:
            log.warning(f"Failed to parse phishing detection: {e}")
            return {
                'phishing_score': 0.5,
                'social_engineering_score': 0.5,
                'urgency_score': 0.5,
                'detected_threat': False,
                'detected_request_for_data': False,
                'explanation': 'LLM parsing failed'
            }

    def _get_fallback_risk_assessment(self) -> Dict[str, Any]:
        """Get fallback risk assessment when LLM fails."""
        return {
            'risk_score': self.fallback['risk_score'],
            'risk_level': self.fallback['risk_level'],
            'risk_category': self.fallback['risk_category'],
            'fraud_pattern': self.fallback['fraud_pattern'],
            'detected_social_engineering': self.fallback['detected_social_engineering'],
            'detected_urgency': self.fallback['detected_urgency'],
            'detected_suspicious_link': self.fallback['detected_suspicious_link'],
            'short_explanation': self.fallback['short_explanation'],
            'risk_tags': self.fallback['risk_tags'],
            'llm_model_name': self.model,
            'llm_model_version': 'fallback',
            'llm_prompt_version': 'v1'
        }