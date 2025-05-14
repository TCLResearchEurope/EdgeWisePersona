from abc import ABC, abstractmethod
from typing import List, Dict

import os
import json
import logging
import torch
from openai import OpenAI
from transformers import pipeline
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

from edgewisepersona.definitions.routine import Routine, RoutinesList


SYSTEM_PROMPT = """
You are an edge-device assistant. From the following conversation history, infer exactly {n_routines} smart-home automation routines 
the user implicitly executes. Order the routines from the one you are most certain about to the one you are least certain about.
Return ONLY JSON valid for the provided schema.

JSON Schema:
{routine_list_schema}
"""


class BaseAIModel(ABC):
    
    def __init__(self):
        self.logger = logging.getLogger()
    
    @abstractmethod
    def build_messages(self, history_text: str, n_routines: int = 10) -> List[Dict[str, str]]:
        """Build the message format required by the model."""
        pass

    @abstractmethod
    def infer_routines(self, history_text: str, n_routines: int = 10, max_routines: int = 10) -> List[Routine]:
        """Infer routines from conversation history."""
        pass

    def parse_model_output(self, response: str, max_routines: int = 10) -> List[Routine]:
        """Parse and validate model output."""
        # Check if response is a list of routines
        try:
            response = json.loads(response)['routines']
        except Exception as e:
            self.logger.error("Failed to parse response as JSON")
            return []
        
        # Check if there are exactly 10 routines
        if len(response) != max_routines:
            self.logger.warning(f"Incorrect number of routines: {len(response)}")
            response = response[:max_routines]

        validated_routines = []
        for routine in response:
            try:
                Routine.model_validate(routine)
                validated_routines.append(routine)
            except Exception as e:
                self.logger.warning("Invalid routine - skipping")
        return validated_routines


class APIModelAdapter(BaseAIModel):
    
    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None):
        super().__init__()
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.model_name = model_name

    def build_messages(self, history_text: str, n_routines: int = 10) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT.format(n_routines=n_routines, routine_list_schema=RoutinesList.model_json_schema())},
            {"role": "user", "content": history_text}
        ]
    
    def infer_routines(self, history_text: str, n_routines: int = 10, max_routines: int = 10) -> List[Routine]:
        messages = self.build_messages(history_text, n_routines)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return self.parse_model_output(response.choices[0].message.content, max_routines)


class LocalModelAdapter(BaseAIModel):
    
    def __init__(self, model_name: str = None, max_new_tokens: int = 8000):
        super().__init__()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.float32 if self.model_name == "google/gemma-3-4b-it" else torch.bfloat16,
            device_map="cpu" if self.model_name == "google/gemma-3-4b-it" else self.device,
        )

        # All local models have a max context window of 128k tokens
        self.generator.tokenizer.model_max_length = 128_000
        self.generator.model.config.max_position_embeddings = 128_000

        # Setup JSON schema parsing
        self.parser = JsonSchemaParser(RoutinesList.model_json_schema())
        self.prefix_function = build_transformers_prefix_allowed_tokens_fn(
            self.generator.tokenizer,
            self.parser
        )

    def build_messages(self, history_text: str, n_routines: int = 10) -> str:
        return self.generator.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT.format(n_routines=n_routines, routine_list_schema=RoutinesList.model_json_schema())},
                {"role": "user", "content": history_text}
            ],
            tokenize=False,
        )
    
    def infer_routines(self, history_text: str, n_routines: int = 10, max_routines: int = 10) -> List[Routine]:
        messages = self.build_messages(history_text, n_routines)
        response = self.generator(
            messages,
            max_new_tokens=self.max_new_tokens,
            prefix_allowed_tokens_fn=self.prefix_function,
            do_sample=False,
            top_k=None,
            top_p=None,
            temperature=None
        )[0]['generated_text'].replace(messages, "")
        
        return self.parse_model_output(response, max_routines)


def get_model(model_name: str, **kwargs) -> BaseAIModel:
    if model_name == "gpt":
        return APIModelAdapter(
            model_name="gpt-4o-2024-11-20", # GPT-4o
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    if model_name == "deepseek":
        return APIModelAdapter(
            model_name="deepseek-chat", # DeepSeek-V3
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    if model_name == "gemini":
        return APIModelAdapter(
            model_name="gemini-2.5-flash-preview-04-17",
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    elif model_name == "qwen":
        return LocalModelAdapter(model_name="Qwen/Qwen2.5-3B-Instruct")
    elif model_name == "llama":
        return LocalModelAdapter(model_name="meta-llama/Llama-3.2-3B-Instruct")
    elif model_name == "phi4":
        return LocalModelAdapter(model_name="microsoft/Phi-4-mini-instruct")
    elif model_name == "gemma3":
        return LocalModelAdapter(model_name="google/gemma-3-4b-it")
    else:
        raise ValueError(f"Unknown model name '{model_name}'")
