from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .roles import Role

class Player:
    _id_counter = 0

    def __init__(self, name: str):
        self.id: int = Player._id_counter
        Player._id_counter += 1
        self.name: str = name
        self.role: 'Role' | None = None
        self.is_alive: bool = True
        self.chat_channels = [] # To be implemented with the chat system
        self.votes_on = 0
        self.voted_for = None
        self.is_on_trial = False
        self.visiting = None
        self.targeted_by = []
        
        # Attributes that can be modified by game mechanics
        self.defense = None # Will be an attribute from Defense enum
        self.is_role_blocked = False
        self.is_doused = False
        self.is_hexed = False
        self.is_blackmailed = False
        self.is_poisoned = False

    def __repr__(self):
        return f"Player({self.name}, {self.role.name if self.role else 'No Role'})"

    def assign_role(self, role: 'Role'):
        self.role = role
        self.defense = role.defense

    def visit(self, target_player: 'Player'):
        self.visiting = target_player

    def clear_night_actions(self):
        self.visiting = None
        self.targeted_by = []
        self.is_role_blocked = False
        if self.role:
            self.defense = self.role.defense # Reset to default defense

    def clear_day_states(self):
        self.votes_on = 0
        self.voted_for = None
        self.is_on_trial = False 