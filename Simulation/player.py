from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .roles import Role
    from .enums import Defense

class Player:
    _id_counter = 0

    def __init__(self, name: str, role: 'Role'):
        self.id: int = Player._id_counter
        Player._id_counter += 1
        self.name: str = name
        self.role: Role = role
        self.is_alive: bool = True
        self.chat_channels = [] #To be implemented with the chat system
        self.votes_on = 0
        self.voted_for = None
        self.is_on_trial = False
        self.visiting: 'Player' = None
        self.targeted_by: list['Player'] = []
        self.protected_by: list['Player'] = []
        
        #Attributes that can be modified by game mechanics
        self.defense: Defense = role.defense
        self.is_role_blocked = False
        self.is_doused = False
        self.is_hexed = False
        self.is_framed = False
        self.is_disguised = False
        self.last_will_bloodied = False
        self.is_jailed = False
        self.is_being_executed = False
        self.is_poisoned = False
        self.poison_timer = 0  #nights until poison kills
        self.notifications = []
        self.is_blackmailed = False
        self.is_controlled = False
        self.is_protected = False
        self.is_healed = False
        self.was_lynched = False
        self.was_stoned = False
        self.poison_uncurable = False
        self.is_infected = False  #Plaguebearer infection
        self.last_will: str = ""  #Player can optionally set a last will.
        #Vote weight for verdict voting (Mayor becomes 3 when revealed)
        self.vote_weight: int = 1

        # Amnesiac – pending role to remember at dawn
        self.remember_role_name = None

        self.was_cleaned = False  # Janitor cleaned corpse flag
        self.was_forged = False   # Forger forged will flag

        # Janitor who cleaned this player (set at night before death resolution)
        self.cleaned_by: 'Player' | None = None

        # Disguiser mechanics
        self.disguised_as_role: 'RoleName' | None = None

        # Jester haunt mechanics
        self.haunt_candidates: list['Player'] | None = None  # set when lynched if Jester
        self.haunt_target: 'Player' | None = None  # chosen target for haunt (night after lynch)

        # Medium séance target (once, when the medium is dead)
        self.seance_target: 'Player' | None = None

    def __repr__(self):
        return f"Player({self.name}, {self.role.name.value})"

    def assign_role(self, role: 'Role'):
        self.role = role
        self.defense = role.defense

    def visit(self, target_player: 'Player'):
        self.visiting = target_player
        target_player.targeted_by.append(self)

    def clear_night_actions(self):
        self.visiting = None
        self.targeted_by = []
        self.protected_by = []
        self.is_role_blocked = False
        if self.role:
            self.defense = self.role.defense #Reset to default defense
        self.is_jailed = False
        self.is_being_executed = False
        self.cleaned_by = None

    def clear_day_states(self):
        self.votes_on = 0
        self.voted_for = None
        self.is_on_trial = False 

    def get_role_name(self) -> str:
        return self.role.name.value

    def can_be_targeted(self, attacker: 'Player') -> bool:
        #This can be expanded with immunities, etc.
        return True

    def reset_night_states(self):
        self.targeted_by = []
        self.visiting = None
        self.protected_by = []
        self.is_role_blocked = False
        self.is_jailed = False
        self.is_being_executed = False
        self.notifications = []
        self.is_blackmailed = False
        self.is_controlled = False
        self.is_poisoned = False
        self.is_doused = False
        self.is_hexed = False
        # Restore default defense (e.g., Veteran loses INVINCIBLE after alert)
        if self.role:
            self.defense = self.role.defense
        #lynch status persists

    def get_public_info(self):
        return {
            "name": self.name,
            "is_alive": self.is_alive
        }