"""
Shared utilities: JSON parsing, confidence intervals, metrics.
"""
import json
import re
import numpy as np
from difflib import SequenceMatcher

try:
    import json5
except ImportError:
    json5 = None


def parse_json_response(text):
    """Multi-stage JSON extraction from LLM output.
    Returns parsed dict or None on failure."""
    if not text:
        return None

    # Stage 1: Extract JSON from markdown code blocks.
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Stage 2: Find the outermost JSON object in the text.
    for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Stage 3: json5 (relaxed parsing) if available.
    if json5 is not None:
        for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL):
            try:
                return json5.loads(match.group())
            except Exception:
                pass

    # Stage 4: Keyword fallback.
    return _keyword_fallback(text)


def _keyword_fallback(text):
    """Extract detection and action from free text using keyword matching."""
    text_lower = text.lower()
    detection = 0
    action = "None"

    emergency_kw = ['call 911', 'emergency room', 'go to the er',
                    'immediate medical attention', 'call 988', 'seek emergency']
    urgent_kw = ['call your doctor', 'contact physician', 'urgent care',
                 'seek medical attention', 'contact doctor', 'see your doctor']
    routine_kw = ['schedule appointment', 'follow up', 'routine',
                  'schedule a visit', 'next appointment']

    if any(kw in text_lower for kw in emergency_kw):
        detection = 1
        action = "Call 911/988"
    elif any(kw in text_lower for kw in urgent_kw):
        detection = 1
        action = "Contact Doctor"
    elif any(kw in text_lower for kw in routine_kw):
        action = "Routine Follow-up"

    return {"detection": detection, "action": action, "evidence": []}


def wilson_ci(count, total, alpha=0.05):
    """Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(count, total, alpha=alpha, method='wilson')
    return (lo, hi)


def bootstrap_ci(y_true, y_pred, metric_func, n_boot=10000, seed=42):
    """Percentile bootstrap 95% CI for a metric function."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            scores.append(metric_func(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    if len(scores) == 0:
        return (np.nan, np.nan)
    return (np.percentile(scores, 2.5), np.percentile(scores, 97.5))


ACTION_MAP = {
    'None': 0, 'Benign': 0,
    'Routine Follow-up': 1, 'Routine': 1,
    'Contact Doctor': 2, 'Urgent': 2,
    'Call 911/988': 3, 'Emergent': 3,
}

ACTION_LABELS = {0: 'None', 1: 'Routine Follow-up', 2: 'Contact Doctor', 3: 'Call 911/988'}


def extract_detection_action(parsed):
    """Extract (detection, action_int) from a parsed JSON dict."""
    if parsed is None:
        return (0, 0)
    det_raw = parsed.get('detection', 0)
    try:
        det = int(det_raw) if det_raw is not None else 0
    except (TypeError, ValueError):
        det = 0
    act_str = str(parsed.get('action', 'None'))
    act = ACTION_MAP.get(act_str, 0)
    return (det, act)


def compute_phantom_symptom_rate(evidence_list, original_message, threshold=0.7):
    """Compute the fraction of claimed findings not grounded in the input.

    Returns (psr, cfs, n_claims) where:
      psr = phantom symptom rate (ungrounded / total)
      cfs = citation fidelity score (valid quotes / total)
      n_claims = total number of evidence items
    """
    if not evidence_list or not original_message:
        return (0.0, 0.0, 0)

    msg_lower = original_message.lower()
    n_claims = len(evidence_list)
    phantoms = 0
    valid_quotes = 0

    for item in evidence_list:
        if not isinstance(item, dict):
            continue
        quote = str(item.get('quote', ''))
        symptom = str(item.get('symptom', ''))

        # Check quote fidelity: does the quoted text appear in the message?
        if quote and len(quote) > 3:
            ratio = SequenceMatcher(None, quote.lower(), msg_lower).find_longest_match(
                0, len(quote.lower()), 0, len(msg_lower)
            ).size / max(len(quote), 1)
            if ratio > threshold:
                valid_quotes += 1
            else:
                # Check if symptom name itself appears.
                if symptom.lower() not in msg_lower:
                    phantoms += 1
        else:
            # No quote provided. Check if symptom name appears.
            if symptom and symptom.lower() not in msg_lower:
                phantoms += 1

    psr = phantoms / n_claims if n_claims > 0 else 0.0
    cfs = valid_quotes / n_claims if n_claims > 0 else 0.0
    return (psr, cfs, n_claims)
