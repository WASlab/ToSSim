from __future__ import annotations

"""Render phase prompts using Jinja templates.

`render_phase_prompt(public_state)` expects a dict with keys:
    day (int)
    phase (str)
    graveyard (list[dict{name,role}])
    votes_needed (int | None)
    vote_board (list[tuple[name, votes]])
    chat_tail (list[str])

It loads `inference/templates/phase_prompt.jinja` if present; otherwise
falls back to a minimal inline template.
"""

from pathlib import Path
import jinja2
from typing import Any

_TEMPLATE_PATH = Path(__file__).parents[1] / "inference" / "templates" / "phase_prompt.jinja"

_env = jinja2.Environment(loader=jinja2.FileSystemLoader(_TEMPLATE_PATH.parent), autoescape=False)

_INLINE = """
Day {{day}} — {{phase|title}}
{% if graveyard %}
Graveyard:
{% for g in graveyard %}• {{g.name}} — {{g.role}}
{% endfor %}{% endif %}
{% if vote_board %}
Nominations ({{votes_needed}} needed):
{% for name, v in vote_board %}• {{name}}: {{v}}
{% endfor %}{% endif %}
{% if chat_tail %}
Recent chat:
{% for line in chat_tail %}{{line}}
{% endfor %}{% endif %}
"""


def render_phase_prompt(state: dict[str, Any]) -> str:
    if _TEMPLATE_PATH.exists():
        template = _env.get_template("phase_prompt.jinja")
        return template.render(**state)
    return jinja2.Template(_INLINE).render(**state) 