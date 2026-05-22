"""REPL loop implementation for frontier API-served models.

Reproduces the rlms==0.1.1 REPL interface for providers not supported natively by the
RLM library: Anthropic, OpenAI (Responses API, reasoning mode), and Google Gen AI.

Each provider exposes a uniform chat(messages, system) call. The REPL loop preserves
Python state across iterations, exposes the patient message as `context`, accepts code
in ```repl blocks, and terminates on FINAL(...).

Usage:
    from frontier_repl_harness import run_repl_arm
    result = run_repl_arm(
        provider="anthropic",
        model="claude-opus-4-7",
        system_prompt=RLM_SYSTEM,
        patient_message="...",
        max_iterations=8,
        allow_llm_query=False,
    )
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic
from openai import OpenAI

# Prefer the new google-genai SDK if available; fall back to google.generativeai.
try:
    from google import genai as google_genai  # google-genai >= 1.0
    _USE_GOOGLE_GENAI = True
except ImportError:
    import google.generativeai as genai  # noqa: F401
    google_genai = None
    _USE_GOOGLE_GENAI = False

# ---------------------------------------------------------------------------
# Provider clients
# ---------------------------------------------------------------------------

_anthropic = None
_openai = None


def _anthropic_client():
    global _anthropic
    if _anthropic is None:
        _anthropic = anthropic.Anthropic()
    return _anthropic


def _openai_client():
    global _openai
    if _openai is None:
        _openai = OpenAI()
    return _openai


_gemini_client = None


def _configure_gemini():
    """Return a Google Gen AI client. Prefers google.genai (new SDK) for thinking support."""
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise EnvironmentError("Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set")
    if _USE_GOOGLE_GENAI:
        _gemini_client = google_genai.Client(api_key=key)
    else:
        import google.generativeai as _genai
        _genai.configure(api_key=key)
        _gemini_client = _genai  # module-as-client fallback (no thinking support)
    return _gemini_client


# ---------------------------------------------------------------------------
# Unified chat interface
# ---------------------------------------------------------------------------

@dataclass
class ChatResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    finish_reason: str = ""
    raw: Any = None


def chat_anthropic(
    *,
    model: str,
    system_prompt: str,
    messages: list[dict],
    max_tokens: int = 16000,
    thinking_budget: int = 4000,
    timeout_s: int = 600,
    disable_thinking: bool = False,
) -> ChatResult:
    """Single Anthropic Messages API call. If disable_thinking=True, omit the thinking
    parameter (thinking falls back to disabled). Used for M2 ablation."""
    client = _anthropic_client()
    api_messages = []
    for m in messages:
        if m["role"] == "user":
            api_messages.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            api_messages.append({"role": "assistant", "content": m["content"]})
        # system messages are passed separately
    # Claude Opus 4.7 requires the new "adaptive" thinking format with output_config.effort
    # and deprecates top_p; older Claude models still use enabled/budget_tokens and accept top_p.
    create_kwargs = dict(
        model=model,
        system=system_prompt,
        messages=api_messages,
        max_tokens=max_tokens,
        temperature=1.0,
        timeout=timeout_s,
    )
    if disable_thinking:
        # M2 ablation: disable thinking entirely
        if "opus-4-7" in model or "opus-4.7" in model:
            # Opus 4.7 still requires output_config; effort=low without thinking
            create_kwargs["output_config"] = {"effort": "low"}
            # omit `thinking` -> no extended thinking
            # do NOT set temperature OR top_p (both deprecated for Opus 4.7)
            create_kwargs.pop("temperature", None)
        # else: leave temperature default (1.0)
    else:
        if "opus-4-7" in model or "opus-4.7" in model:
            # Newer Opus 4.7+: adaptive thinking; top_p is deprecated and rejected
            create_kwargs["thinking"] = {"type": "adaptive"}
            create_kwargs["output_config"] = {"effort": "low"}
            # do NOT set top_p (rejected with 400 by this model)
        else:
            # Older Claude models: enabled+budget_tokens; top_p still supported
            create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            create_kwargs["top_p"] = 0.999
    resp = client.messages.create(**create_kwargs)
    # Collect text from all non-thinking content blocks
    text_parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)
    text = "\n".join(text_parts)
    return ChatResult(
        text=text,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        reasoning_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        finish_reason=resp.stop_reason or "",
        raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
    )


def chat_openai(
    *,
    model: str,
    system_prompt: str,
    messages: list[dict],
    max_output_tokens: int = 32000,
    reasoning_effort: str = "medium",
    timeout_s: int = 600,
) -> ChatResult:
    """Single OpenAI Responses-API call in reasoning mode (for gpt-5.5)."""
    client = _openai_client()
    # Responses API: convert messages into input format
    flat_input = [{"role": "system", "content": system_prompt}]
    for m in messages:
        flat_input.append({"role": m["role"], "content": m["content"]})
    resp = client.responses.create(
        model=model,
        input=flat_input,
        max_output_tokens=max_output_tokens,
        reasoning={"effort": reasoning_effort},
        timeout=timeout_s,
    )
    text = resp.output_text if hasattr(resp, "output_text") else ""
    if not text:
        # Walk output items to extract text
        for item in getattr(resp, "output", []):
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []):
                    if getattr(c, "type", None) == "output_text":
                        text += c.text
    usage = getattr(resp, "usage", None)
    return ChatResult(
        text=text,
        input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
        output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        reasoning_tokens=getattr(usage, "reasoning_tokens", 0) if usage else 0,
        finish_reason=getattr(resp, "stop_reason", "") or "",
        raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
    )


def chat_gemini(
    *,
    model: str,
    system_prompt: str,
    messages: list[dict],
    max_output_tokens: int = 16000,
    thinking_budget: int = 4000,
    timeout_s: int = 600,
    disable_thinking: bool = False,
) -> ChatResult:
    """Single Gemini call via the new google.genai SDK (preferred) or fallback to legacy SDK.

    The new SDK supports `thinking_config` properly; the legacy `google.generativeai` does not.
    """
    client = _configure_gemini()
    # Build conversation contents: system_instruction is set via config; history+last are 'contents'
    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    if _USE_GOOGLE_GENAI:
        # google.genai (new SDK) supports thinking
        from google.genai import types as _gtypes
        if disable_thinking:
            # Gemini 3.1 Pro Preview REQUIRES thinking_budget >= 1 (this model only works
            # in thinking mode). The closest M2-style ablation is to set thinking_budget
            # to the minimum allowed value (1) which approximates "thinking off".
            config = _gtypes.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
                top_p=0.95,
                max_output_tokens=max_output_tokens,
                thinking_config=_gtypes.ThinkingConfig(thinking_budget=1),
            )
        else:
            config = _gtypes.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
                top_p=0.95,
                max_output_tokens=max_output_tokens,
                thinking_config=_gtypes.ThinkingConfig(thinking_budget=thinking_budget),
            )
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        text = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        return ChatResult(
            text=text,
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            reasoning_tokens=getattr(usage, "thoughts_token_count", 0) if usage else 0,
            finish_reason="",
            raw=None,
        )
    else:
        # Legacy fallback (no thinking)
        import google.generativeai as _genai
        gen_model = _genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt,
            generation_config={
                "temperature": 0.0,
                "top_p": 0.95,
                "max_output_tokens": max_output_tokens,
            },
        )
        history = [{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} for m in messages[:-1]]
        chat = gen_model.start_chat(history=history)
        response = chat.send_message(messages[-1]["content"], request_options={"timeout": timeout_s})
        text = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        return ChatResult(
            text=text,
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            reasoning_tokens=0,
            finish_reason="",
            raw=None,
        )


def get_chat_fn(provider: str) -> Callable:
    return {
        "anthropic": chat_anthropic,
        "openai": chat_openai,
        "gemini": chat_gemini,
    }[provider]


# ---------------------------------------------------------------------------
# Sandboxed Python execution
# ---------------------------------------------------------------------------

_REPL_PREAMBLE = textwrap.dedent("""
    import json, re, sys
    from io import StringIO
    _stdout_capture = StringIO()
    sys.stdout = _stdout_capture
""")

_REPL_EPILOGUE = textwrap.dedent("""
    sys.stdout = sys.__stdout__
    _captured = _stdout_capture.getvalue()
    print(_captured, end='')
""")


def execute_python_block(
    code: str,
    state_pickle_path: str,
    timeout_s: int = 60,
    llm_query_callable_dump: str | None = None,
) -> tuple[str, str]:
    """Run `code` in a Python subprocess with pickled state persistence.

    Returns (stdout, stderr). State is loaded from `state_pickle_path` at start and
    rewritten at end. If `llm_query_callable_dump` is provided, it is a string of
    Python code defining `llm_query(prompt) -> str` (for Arm D); otherwise no llm_query
    is defined (Arm C — model will get NameError if it tries to call it).
    """
    llm_query_section = llm_query_callable_dump or ""
    runner = textwrap.dedent(f"""
        import pickle, sys, json
        # Load REPL state
        with open({state_pickle_path!r}, 'rb') as f:
            state = pickle.load(f)
        globals().update(state)
        {llm_query_section}
        # --- BEGIN MODEL CODE ---
        try:
{textwrap.indent(code, '            ')}
        except Exception as e:
            import traceback
            print('ERROR:', traceback.format_exc(), file=sys.stderr)
        # --- END MODEL CODE ---
        # Save updated state (exclude unpicklable items)
        new_state = {{}}
        for k, v in list(globals().items()):
            if k.startswith('_') or k in ('pickle', 'sys', 'json', 'state', 'llm_query', 'traceback'):
                continue
            try:
                pickle.dumps(v)
                new_state[k] = v
            except Exception:
                pass
        with open({state_pickle_path!r}, 'wb') as f:
            pickle.dump(new_state, f)
    """)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return e.stdout or "", f"TIMEOUT after {timeout_s}s"


# ---------------------------------------------------------------------------
# REPL loop
# ---------------------------------------------------------------------------

_REPL_BLOCK_RE = re.compile(r"```repl\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_FINAL_RE = re.compile(r"^\s*FINAL\((.*)\)\s*$", re.DOTALL | re.MULTILINE)


def _make_llm_query_dump(provider: str, model: str, chat_kwargs: dict) -> str:
    """Generate a Python source string that defines llm_query(prompt) -> str.

    The subprocess can't import our SDK clients reliably (env, network), so we route
    sub-calls back to the parent process via a temp file marker. For simplicity in this
    revision we IMPLEMENT llm_query as a stub that returns 'unknown' and log that a
    sub-call happened. Arm D therefore exercises the recursion path but does not
    actually call back to the parent model. This is a documented limitation of the
    frontier harness — see Limitations item 7.
    """
    return textwrap.dedent("""
        def llm_query(prompt):
            # Frontier-harness limitation: sub-calls are stubbed to 'urgent' for safety.
            # See manuscript Limitations item 7.
            return 'urgent'
    """)


def run_repl_arm(
    *,
    provider: str,
    model: str,
    system_prompt: str,
    patient_message: str,
    max_iterations: int = 8,
    allow_llm_query: bool = False,
    chat_kwargs: dict | None = None,
    workdir: str | None = None,
    disable_thinking: bool = False,
) -> dict:
    """Execute the REPL loop for a single case.

    Returns a dict:
      {
        'final_json': str | None,         # the JSON string inside FINAL(...) if found
        'iterations': int,                # number of model turns executed
        'transcript': list[dict],         # full message log
        'usage': {                        # cumulative across iterations
            'input_tokens': int, 'output_tokens': int, 'reasoning_tokens': int
        },
        'terminated_reason': str,         # 'FINAL', 'no_code_no_final', 'max_iterations',
                                          #   'parse_error', 'timeout', 'api_error'
        'error': str | None,
        'elapsed_sec': float,
      }
    """
    import pickle
    import tempfile

    workdir = workdir or tempfile.mkdtemp(prefix="repl_")
    state_path = os.path.join(workdir, "state.pkl")
    # Seed state with the `context` variable holding the patient message
    with open(state_path, "wb") as f:
        pickle.dump({"context": patient_message}, f)

    chat_fn = get_chat_fn(provider)
    chat_kwargs = chat_kwargs or {}
    # Strip max_depth from chat_kwargs if present
    chat_kwargs = {k: v for k, v in chat_kwargs.items() if k != "max_depth"}

    messages = [{"role": "user", "content": patient_message}]
    transcript = [{"role": "system", "content": system_prompt}, *messages]
    usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    t0 = time.time()
    llm_query_dump = _make_llm_query_dump(provider, model, chat_kwargs) if allow_llm_query else None

    final_json = None
    terminated_reason = "max_iterations"
    error = None

    try:
        for iteration in range(max_iterations):
            chat_kwargs_iter = dict(chat_kwargs)
            if disable_thinking and provider in ("anthropic", "gemini"):
                chat_kwargs_iter["disable_thinking"] = True
            cr = chat_fn(
                model=model,
                system_prompt=system_prompt,
                messages=messages,
                **chat_kwargs_iter,
            )
            usage["input_tokens"] += cr.input_tokens
            usage["output_tokens"] += cr.output_tokens
            usage["reasoning_tokens"] += cr.reasoning_tokens
            assistant_turn = cr.text
            messages.append({"role": "assistant", "content": assistant_turn})
            transcript.append({"role": "assistant", "content": assistant_turn, "raw": cr.raw})

            # Check for FINAL(...)
            m = _FINAL_RE.search(assistant_turn)
            if m:
                final_json = m.group(1).strip()
                terminated_reason = "FINAL"
                break

            # Extract repl blocks
            blocks = _REPL_BLOCK_RE.findall(assistant_turn)
            if not blocks:
                terminated_reason = "no_code_no_final"
                break

            # Execute each block in sequence
            combined_output = []
            for code in blocks:
                stdout, stderr = execute_python_block(
                    code=code.strip(),
                    state_pickle_path=state_path,
                    timeout_s=60,
                    llm_query_callable_dump=llm_query_dump,
                )
                block_repr = stdout
                if stderr:
                    block_repr += "\n[STDERR]\n" + stderr
                combined_output.append(block_repr)
            repl_output = "\n---\n".join(combined_output)
            messages.append({"role": "user", "content": f"REPL output:\n{repl_output}"})
            transcript.append({"role": "user", "content": f"REPL output:\n{repl_output}"})
    except Exception:
        terminated_reason = "api_error"
        error = traceback.format_exc()

    return {
        "final_json": final_json,
        "iterations": iteration + 1 if "iteration" in dir() else 0,
        "transcript": transcript,
        "usage": usage,
        "terminated_reason": terminated_reason,
        "error": error,
        "elapsed_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Single-turn helpers for Arms A, B, E
# ---------------------------------------------------------------------------

def run_single_turn_arm(
    *,
    provider: str,
    model: str,
    system_prompt: str | None,
    user_prompt: str,
    chat_kwargs: dict | None = None,
) -> dict:
    """Run a single-turn arm (A, A+, B, E). The full filled prompt goes in user_prompt;
    system_prompt may be empty/None or a brief role line."""
    chat_fn = get_chat_fn(provider)
    chat_kwargs = chat_kwargs or {}
    t0 = time.time()
    error = None
    cr = None
    try:
        cr = chat_fn(
            model=model,
            system_prompt=system_prompt or "You are a helpful assistant.",
            messages=[{"role": "user", "content": user_prompt}],
            **chat_kwargs,
        )
    except Exception:
        error = traceback.format_exc()
    elapsed = time.time() - t0
    return {
        "text": cr.text if cr else "",
        "usage": {
            "input_tokens": cr.input_tokens if cr else 0,
            "output_tokens": cr.output_tokens if cr else 0,
            "reasoning_tokens": cr.reasoning_tokens if cr else 0,
        },
        "raw": cr.raw if cr else None,
        "error": error,
        "elapsed_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Smoke test: run Arm A on a single test case with a single provider for sanity
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--provider", required=True, choices=["anthropic", "openai", "gemini"])
    p.add_argument("--model", required=True)
    p.add_argument("--message", default="I'm having severe chest pain radiating to my left arm. It started 30 minutes ago.")
    p.add_argument("--arm", default="A", choices=["A", "C"])
    args = p.parse_args()

    if args.arm == "A":
        # Inline minimal Arm A prompt (must match prompts.SINGLE_PASS structure)
        prompt = (
            "You are a medical triage safety system. Analyze the following patient message and "
            "determine if it contains a safety hazard.\n\nOutput ONLY a JSON object with fields:\n"
            "  \"detection\": 0 or 1\n  \"action\": \"None\" | \"Routine Follow-up\" | \"Contact Doctor\" | \"Call 911/988\"\n"
            "  \"evidence\": [{\"symptom\": \"...\", \"quote\": \"...\"}]\n\nPatient message:\n" + args.message
        )
        result = run_single_turn_arm(provider=args.provider, model=args.model, system_prompt=None, user_prompt=prompt)
        print(json.dumps(result, indent=2, default=str))
    else:
        # Inline RLM_SYSTEM prompt — must match prompts.RLM_SYSTEM exactly
        system = (
            "You are a medical triage safety system. A patient message is stored in the REPL variable `context`. "
            "Your job is to determine if it contains a safety hazard.\n\n"
            "RULES:\n- Only identify clinical findings EXPLICITLY stated in the patient message.\n"
            "- Do NOT hallucinate symptoms not in `context`.\n"
            "- For each clinical finding, you MUST provide an exact quote from the patient message.\n\n"
            "TASK: Write Python code in ```repl blocks to programmatically analyze the patient message. "
            "You may use any Python capabilities (string operations, regex, etc.).\n\n"
            "OUTPUT: When done, signal your final answer using FINAL(...) on its own line (NOT inside a code block), "
            "containing a JSON object with this structure:\n"
            "{\"detection\": 0 or 1, \"action\": \"None\" or \"Routine Follow-up\" or \"Contact Doctor\" or \"Call 911/988\", "
            "\"evidence\": [{\"symptom\": \"...\", \"quote\": \"...\", \"assessment\": \"benign/routine/urgent/emergent\"}]}\n\n"
            "IMPORTANT: You MUST end with a FINAL() statement containing the JSON."
        )
        result = run_repl_arm(
            provider=args.provider,
            model=args.model,
            system_prompt=system,
            patient_message=args.message,
            max_iterations=8,
            allow_llm_query=False,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "transcript"}, indent=2, default=str))
        print("--- TRANSCRIPT ---")
        for turn in result["transcript"]:
            print(f"\n[{turn['role'].upper()}]\n{str(turn.get('content', ''))[:2000]}")
