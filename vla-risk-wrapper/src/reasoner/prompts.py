from __future__ import annotations

SYSTEM_PROMPT = """\
You are a manipulation failure analyzer for a robotic arm performing contact-rich tasks.
Given a current RGB image, a task instruction, the last 8 actions, and a risk score,
you must select exactly ONE recovery primitive from this list:
  - re_grasp: re-attempt the grasp
  - re_approach: retreat and re-approach the target
  - align_then_insert: correct yaw alignment then slowly insert
  - request_help: stop and signal for human assistance
  - continue: no recovery needed, continue normally

IMPORTANT: You MUST output ONLY valid JSON with exactly this format and nothing else:
{"primitive": "<one of the five IDs above>", "rationale": "<one short sentence>"}

Do not include any explanation outside the JSON. Do not wrap in code blocks.\
"""

USER_TEMPLATE = (
    "Instruction: {instruction}\n"
    "Risk score: {risk:.3f}\n"
    "Last {n_actions} actions (dx,dy,dz,drx,dry,drz,gripper):\n{actions_str}\n"
    "Based on the image and above context, select the best recovery primitive."
)


def build_user_message(
    instruction: str,
    recent_actions,
    risk_score: float,
) -> str:
    import numpy as np

    actions_arr = np.asarray(recent_actions)
    lines = [f"  [{i}]: " + ", ".join(f"{v:.3f}" for v in row) for i, row in enumerate(actions_arr)]
    actions_str = "\n".join(lines)

    return USER_TEMPLATE.format(
        instruction=instruction,
        risk=risk_score,
        n_actions=len(actions_arr),
        actions_str=actions_str,
    )
