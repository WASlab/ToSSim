from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from .enums import RoleName, Faction, RoleAlignment, Attack, Defense, Priority

if TYPE_CHECKING:
    from .player import Player

class Role(ABC):
    def __init__(self):
        self.name: RoleName = None
        self.faction: Faction = None
        self.alignment: RoleAlignment = None
        self.attack: Attack = Attack.NONE
        self.defense: Defense = Defense.NONE
        self.is_unique: bool = False
        self.action_priority = Priority.MISC_NON_LETHAL

    @abstractmethod
    def perform_night_action(self, player: 'Player', target: 'Player' = None):
        pass

    def get_info(self):
        return {
            "name": self.name.value,
            "faction": self.faction.name,
            "alignment": self.alignment.name,
            "attack": self.attack.name,
            "defense": self.defense.name,
            "is_unique": self.is_unique
        }

# Town Roles
class Sheriff(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.SHERIFF
        self.faction = Faction.TOWN
        self.alignment = RoleAlignment.TOWN_INVESTIGATIVE

    def perform_night_action(self, player: 'Player', target: 'Player'):
        # Updated to handle detection immunity
        if target.role.name == RoleName.GODFATHER:
            return f"{target.name} is not suspicious."
        if target.role.faction == Faction.MAFIA:
            return f"{target.name} is suspicious!"
        if target.role.name == RoleName.SERIAL_KILLER:
            return f"{target.name} is suspicious!"
        # This is a simplification. Coven, other immune roles, etc. needs to be added.
        return f"{target.name} is not suspicious."

class Investigator(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.INVESTIGATOR
        self.faction = Faction.TOWN
        self.alignment = RoleAlignment.TOWN_INVESTIGATIVE

    def perform_night_action(self, player: 'Player', target: 'Player'):
        # Simplified investigation results.
        # A full implementation would have result groups.
        return f"Your target could be a {target.role.name.value}."

class Lookout(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.LOOKOUT
        self.faction = Faction.TOWN
        self.alignment = RoleAlignment.TOWN_INVESTIGATIVE

    def perform_night_action(self, player: 'Player', target: 'Player'):
        visitors = [p.name for p in target.targeted_by]
        if not visitors:
            return f"No one visited {target.name}."
        return f"These players visited {target.name}: {', '.join(visitors)}"

class Spy(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.SPY
        self.faction = Faction.TOWN
        self.alignment = RoleAlignment.TOWN_INVESTIGATIVE

    def perform_night_action(self, player: 'Player', target: 'Player' = None):
        # In this simulation, Spy will be implemented to see who Mafia and Coven visit.
        # This requires access to the game state, which will be handled in the Game class.
        return "Spy action needs game state access."

class Psychic(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.PSYCHIC
        self.faction = Faction.TOWN
        self.alignment = RoleAlignment.TOWN_INVESTIGATIVE

    def perform_night_action(self, player: 'Player', target: 'Player' = None):
        # This also requires game state access to get visions.
        return "Psychic action needs game state access."

class Tracker(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.TRACKER
        self.faction = Faction.TOWN
        self.alignment = RoleAlignment.TOWN_INVESTIGATIVE

    def perform_night_action(self, player: 'Player', target: 'Player'):
        if target.visiting:
            return f"{target.name} visited {target.visiting.name}."
        return f"{target.name} did not visit anyone."

# Town Protective
class Doctor(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.DOCTOR
        self.faction = Faction.TOWN
        self.alignment = RoleAlignment.TOWN_PROTECTIVE
        self.self_heals = 1

    def perform_night_action(self, player: 'Player', target: 'Player'):
        if player == target:
            if self.self_heals > 0:
                self.self_heals -= 1
                return f"You healed yourself. You have {self.self_heals} self-heals remaining."
            else:
                return "You are out of self-heals."
        return f"You are healing {target.name} tonight."

class Bodyguard(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.BODYGUARD
        self.faction = Faction.TOWN
        self.alignment = RoleAlignment.TOWN_PROTECTIVE
        self.vests = 1

    def perform_night_action(self, player: 'Player', target: 'Player'):
        if player == target:
            if self.vests > 0:
                self.vests -= 1
                player.defense = Defense.BASIC
                return f"You are guarding yourself tonight. You have {self.vests} vests remaining."
            else:
                return "You are out of vests."
        return f"You are guarding {target.name} tonight."

# Town Killing
class Vigilante(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.VIGILANTE
        self.faction = Faction.TOWN
        self.alignment = RoleAlignment.TOWN_KILLING
        self.attack = Attack.BASIC
        self.bullets = 3

    def perform_night_action(self, player: 'Player', target: 'Player'):
        if self.bullets > 0:
            self.bullets -= 1
            return f"You are shooting {target.name} tonight. You have {self.bullets} bullets remaining."
        else:
            return "You are out of bullets."

# Town Support
class Mayor(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.MAYOR
        self.faction = Faction.TOWN
        self.alignment = RoleAlignment.TOWN_SUPPORT
        self.is_unique = True
        self.revealed = False

    def reveal(self, player):
        if not self.revealed:
            self.revealed = True
            player.votes = 3
            print(f"{player.name} has revealed themselves as the Mayor!")
            # Cannot be healed after reveal - this will be handled in the game logic.
            return True
        return False
    
    def perform_night_action(self, player: 'Player', target: 'Player' = None):
        # Mayor has no night action
        return None

class Medium(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.MEDIUM
        self.faction = Faction.TOWN
        self.alignment = RoleAlignment.TOWN_SUPPORT
        self.seances = 1

    def perform_night_action(self, player: 'Player', target: 'Player' = None):
        # Logic for speaking with the dead will be in the chat manager
        return "You are speaking with the dead."
    
    def seance(self, living_player):
        if self.seances > 0:
            self.seances -= 1
            # The actual message sending will be handled in the game/chat logic
            return f"You can send a message to {living_player.name} from beyond the grave."
        return "You are out of seances."

class Escort(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.ESCORT
        self.faction = Faction.TOWN
        self.alignment = RoleAlignment.TOWN_SUPPORT

    def perform_night_action(self, player: 'Player', target: 'Player'):
        target.is_role_blocked = True
        return f"You are distracting {target.name} tonight."

# Mafia Roles
class Godfather(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.GODFATHER
        self.faction = Faction.MAFIA
        self.alignment = RoleAlignment.MAFIA_KILLING
        self.attack = Attack.BASIC
        self.defense = Defense.BASIC
        self.is_unique = True

    def perform_night_action(self, player: 'Player', target: 'Player'):
        # The Godfather orders the Mafioso to kill. If no Mafioso, they kill themselves.
        # This logic is handled in the game loop.
        print(f"{player.name} (Godfather) is ordering a hit on {target.name}.")
        return None

class Mafioso(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.MAFIOSO
        self.faction = Faction.MAFIA
        self.alignment = RoleAlignment.MAFIA_KILLING
        self.attack = Attack.BASIC

    def perform_night_action(self, player: 'Player', target: 'Player'):
        # The actual killing logic will be handled by the game loop
        # based on the attack value of the role.
        # This function can return information or set states.
        print(f"{player.name} (Mafioso) is attacking {target.name}.")
        return None

# Neutral Roles
class SerialKiller(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.SERIAL_KILLER
        self.faction = Faction.NEUTRAL
        self.alignment = RoleAlignment.NEUTRAL_KILLING
        self.attack = Attack.BASIC
        self.defense = Defense.BASIC
    
    def perform_night_action(self, player: 'Player', target: 'Player'):
        print(f"{player.name} (Serial Killer) is attacking {target.name}.")
        # If roleblocked, kill the roleblocker. This logic will be in the game loop.
        return None 
