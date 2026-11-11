from openai import OpenAI
import requests
import time
import re
from core.utils import log, secs
import sys

class ChatVLLM:
    
    def __init__(self, config):
        self.client = OpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
        )
        self.model = config["model_name"]
        self.temperature =  config["temperature"]
        self.enable_thinking = config["enable_thinking"]
        self.tokenize_url = config["base_url"].rstrip("/").removesuffix("/v1") + "/tokenize"

    def _count_tokens(self, text):
        try:
            r = requests.post(self.tokenize_url, json={"model": self.model, "prompt": text}, timeout=10)
            r.raise_for_status()
            return r.json()["count"]
        except Exception:
            return None
    
    def invoke(self, prompt, variables=None, task=None, **log_kv):
        content = prompt.format(**variables) if variables else prompt
    
        task = task or sys._getframe(1).f_code.co_name
        t0 = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": content},
                ],
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
                },
            )
        except Exception as e:
            log("error", "llm", level="ERROR", task=task, **log_kv, model=self.model,
                thinking=str(self.enable_thinking).lower(),
                error=f"{type(e).__name__}: {e}", time=secs(time.perf_counter() - t0))
            raise
        elapsed = time.perf_counter() - t0

        message = response.choices[0].message
        text = message.content or ""
        if self.enable_thinking:
            text = self._strip_thinking(text)

        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
        tps = completion_tokens / elapsed if elapsed > 0 else 0

        # cached: needs vLLM --enable-prompt-tokens-details, otherwise null
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(prompt_details, "cached_tokens", None)

        # reasoning: from usage if the server reports it, else completion - tokens(answer)
        reasoning = None
        if self.enable_thinking:
            completion_details = getattr(usage, "completion_tokens_details", None)
            reasoning = getattr(completion_details, "reasoning_tokens", None)
            if reasoning is None:
                answer_tokens = self._count_tokens(text)
                if answer_tokens is not None:
                    reasoning = max(completion_tokens - answer_tokens, 0)

        log("ok", "llm",
            task=task,
            **log_kv,
            model=self.model,
            thinking=str(self.enable_thinking).lower(),
            prompt=prompt_tokens,
            cached=cached,
            completion=completion_tokens,
            reasoning=reasoning if self.enable_thinking else 0,
            total=total_tokens,
            tps=f"{tps:.1f}",
            time=secs(elapsed),
        )
        return text
    
    @staticmethod
    def _strip_thinking(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        return text.strip()


def create_llm(config):
    return ChatVLLM(config)
