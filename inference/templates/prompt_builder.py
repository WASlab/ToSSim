from pathlib import Path
import random
from dataclasses import dataclass, field
import jinja2

# Optional – if a real registry module exists it will be used. Otherwise fall back to an empty dict.
try:
    from Simulation.tools.registry import get_tool_registry  # type: ignore
except (ImportError, ModuleNotFoundError):
    def get_tool_registry():  # pragma: no cover
        return {}

from Simulation.enums import RoleName  # For type hints only – avoids heavy Role import

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
_env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATE_DIR), autoescape=False)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RoleCard:
    """A minimal, serialisable representation of a role's public card.

    This is intentionally lightweight so it can be passed straight into a
    Jinja template without additional pre-processing.
    """

    # Core identity
    name: str
    faction: str
    alignment: str
    goal: str = "Win condition unknown (placeholder)."

    # Combat stats
    attack: str = "None"
    defense: str = "None"
    immunities: list[str] | None = field(default_factory=list)
    is_unique: bool = False

    # Gameplay text – kept short to conserve prompt tokens
    abilities: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    night_action_usage: str = "Use your role-specific night action tool when appropriate."
    wins_with: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a plain dict that Jinja can iterate over easily."""
        return {
            **self.__dict__,
            # Convert booleans/lists to a friendly representation
            "is_unique": "Yes" if self.is_unique else "No",
        }

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_DEFAULT_NAMES = [
    "Alice",
    "Bob",
    "Charlie",
    "Daphne",
    "Eve",
    "Mallory",
    "Oscar",
    "Peggy",
    "Sybil",
    "Trent",
    "Victor",
    "Walter",
]


def generate_agent_name() -> str:
    """Return a pseudo-random agent display name.

    The simulator will eventually replace this with its own name-source.  Until
    then we draw from a fixed sample list so that unit tests are repeatable.
    """
    return random.choice(_DEFAULT_NAMES)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_role_card(role: "Role") -> RoleCard:  # type: ignore
    """Construct a *starter* RoleCard from a Role instance.

    The function is deliberately conservative: it copies whatever static info
    is already available on the runtime `Role` object.  Missing narrative
    fields fall back to placeholder text so that the agent prompt remains
    syntactically valid even if the content is incomplete.
    """
    info = role.get_info() if hasattr(role, "get_info") else {}

    return RoleCard(
        name=info.get("name", getattr(role, "name", "Unknown")),
        faction=info.get("faction", getattr(role, "faction", "Unknown")),
        alignment=info.get("alignment", getattr(role, "alignment", "Unknown")),
        attack=info.get("attack", getattr(role, "attack", "None")),
        defense=info.get("defense", getattr(role, "defense", "None")),
        is_unique=info.get("is_unique", getattr(role, "is_unique", False)),
    )


def build_system_prompt(agent_name: str, role_card: RoleCard) -> str:
    """Render the system prompt string given an agent name and RoleCard."""

    # Ensure the template file exists; if not, fall back to an inline template.
    template_name = "system_prompt.jinja"
    if (TEMPLATE_DIR / template_name).exists():
        template = _env.get_template(template_name)
        return template.render(agent_name=agent_name, rc=role_card.to_dict())

    # Inline minimal template (used during initial bootstrapping).
    inline_tpl = (
        """
<system>
You are {{agent_name}}, the {{rc.name}}.

Alignment: {{rc.alignment}}
Goal: {{rc.goal}}

Abilities:{% for abl in rc.abilities %}
• {{abl}}{% endfor %}{% if not rc.abilities %}
• (None listed yet){% endif %}

Attributes:{% for attr in rc.attributes %}
• {{attr}}{% endfor %}{% if not rc.attributes %}
• (None listed yet){% endif %}

Night Action:
{{rc.night_action_usage}}

Wins With: {{rc.wins_with | join(', ') if rc.wins_with else 'Unknown'}}
Unique: {{rc.is_unique}}

Guidelines:
• Think privately inside <think>…</think>.
• Call tools with a single XML tag (e.g., <get_role>Doctor</get_role>).
• End your turn with one of <speak>, <whisper>, <vote>, or <wait>.
</system>
"""
    )
    return jinja2.Template(inline_tpl).render(agent_name=agent_name, rc=role_card.to_dict())