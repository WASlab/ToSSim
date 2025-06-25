from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from .enums import RoleName, Faction, RoleAlignment, Attack, Defense, Priority, DisplayName, VisitType, ImmunityType
from .alignment import get_role_alignment, get_role_faction
import random
from .enums import DuelMove, DuelDefense

if TYPE_CHECKING:
    from .player import Player
    from .game import Game

class Role(ABC):
    def __init__(self):
        self.name: RoleName = None
        self.faction: Faction = None
        self.alignment: RoleAlignment = None
        self.attack: Attack = Attack.NONE
        self.defense: Defense = Defense.NONE
        self.is_unique: bool = False
        self.action_priority = Priority.SUPPORT_DECEPTION
        self.is_roleblock_immune = False
        self.detection_immune = False
        self.control_immune = False
        #Visit type: default harmful if attacking, else non-harmful
        self.visit_type = VisitType.HARMFUL if self.attack != Attack.NONE else VisitType.NON_HARMFUL
        self.is_coven = False
        self.has_necronomicon = False

    @abstractmethod
    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        pass

    def perform_day_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
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

#utility

def has_immunity(role: 'Role', immunity: ImmunityType) -> bool:
    if immunity == ImmunityType.ROLE_BLOCK:
        return getattr(role, 'is_roleblock_immune', False)
    if immunity == ImmunityType.CONTROL:
        return getattr(role, 'control_immune', False)
    if immunity == ImmunityType.DETECTION:
        return getattr(role, 'detection_immune', False)
    return False

#Town Roles
class Sheriff(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.SHERIFF
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.INVESTIGATION

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game'):
        if target.role.detection_immune:
            return f"{target.name} is not suspicious."
        if target.role.name == RoleName.SERIAL_KILLER:
            return f"{target.name} is suspicious!"
        if target.role.faction == Faction.MAFIA and target.role.name != RoleName.GODFATHER:
            return f"{target.name} is suspicious!"
        return f"{target.name} is not suspicious."

class Investigator(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.INVESTIGATOR
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.INVESTIGATION

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game'):
        if not game or not target:
            return "You must select a target."
        investigated_role = target.role.name
        if target.is_framed or (hasattr(game, 'config') and game.config.is_coven and target.is_hexed):
            investigated_role = RoleName.HEX_MASTER if game.config.is_coven else RoleName.FRAMER
        elif target.is_doused:
            investigated_role = RoleName.ARSONIST
        
        if hasattr(game, 'config'):
            result_group = game.config.get_investigator_result_group(investigated_role)
            if not result_group:
                return f"Your target, {target.name}, has a role that is mysterious and unknown."
            role_names = [role.value for role in result_group]
            if len(role_names) == 1:
                return f"Your target is a {role_names[0]}."
            result_string = ", ".join(role_names[:-1]) + ", or " + role_names[-1]
            return f"Your target could be a {result_string}."
        
        #Fallback if no config is present
        return f"Your target is a {investigated_role.value}."

class Lookout(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.LOOKOUT
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.INVESTIGATION

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game'):
        from .enums import VisitType
        visitors = [p.name for p in target.targeted_by if p.role.visit_type != VisitType.ASTRAL]
        if not visitors:
            return f"No one visited {target.name}."
        return f"These players visited {target.name}: {', '.join(visitors)}"

class Doctor(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.DOCTOR
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.CONTROL_PROTECTION
        self.self_heals = 1

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not target:
            return None
        if hasattr(target.role, 'revealed') and getattr(target.role, 'revealed', False):
            return "You cannot heal a revealed Mayor."
        if player == target:
            if self.self_heals > 0:
                self.self_heals -= 1
                target.defense = Defense.POWERFUL
                return f"You are using a self-heal. You have {self.self_heals} left."
            else:
                return "You are out of self-heals."
        target.defense = Defense.POWERFUL
        return f"You are healing {target.name} tonight."

class Bodyguard(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.BODYGUARD
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.CONTROL_PROTECTION
        self.vests = 1

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game'):
        if player == target:
            if self.vests > 0:
                self.vests -= 1
                player.defense = Defense.BASIC
                return f"You are guarding yourself tonight. You have {self.vests} vests remaining."
            else:
                return "You are out of vests."
        if target:
            target.protected_by.append(player)
            return f"You are guarding {target.name} tonight."
        return None

class Vigilante(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.VIGILANTE
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.attack = Attack.BASIC
        self.action_priority = Priority.KILLING
        self.bullets = 3
        self.has_killed_townie = False
        self.put_gun_down = False
        self.visit_type = VisitType.HARMFUL

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not target or not game:
            return None
        
        if self.bullets > 0:
            self.bullets -= 1
            game.register_attack(player, target, self.attack)
            return f"You are shooting {target.name} tonight. You have {self.bullets} bullets remaining."
        else:
            return "You are out of bullets."

class Mayor(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.MAYOR
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_unique = True
        self.revealed = False

    def reveal(self, player):
        if not self.revealed:
            self.revealed = True
            player.votes = 3
            print(f"{player.name} has revealed themselves as the Mayor!")
            return True
        return False
    
    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        return None

class Medium(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.MEDIUM
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.HIGHEST
        self.seances = 1

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        return "You are speaking with the dead."
    
    def seance(self, living_player):
        if self.seances > 0:
            self.seances -= 1

class TavernKeeper(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.TAVERN_KEEPER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_roleblock_immune = True
        self.action_priority = Priority.CONTROL_PROTECTION

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not target:
            return None
        if target.role.name == RoleName.SERIAL_KILLER and not target.role.cautious:
            return None
        if has_immunity(target.role, ImmunityType.ROLE_BLOCK):
            if isinstance(target.role, SerialKiller) and target.role.cautious:
                 target.notifications.append("Someone tried to role block you, but you are immune!")
            return f"{target.name} could not be role-blocked."
        target.is_role_blocked = True
        return f"You are distracting {target.name} tonight."

class Godfather(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.GODFATHER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.defense = Defense.BASIC
        self.is_unique = True
        self.detection_immune = True
        self.action_priority = Priority.KILLING
        self.is_roleblock_immune = False 
        self.visit_type = VisitType.HARMFUL

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        mafioso = game.find_player_by_role(RoleName.MAFIOSO)
        if not mafioso or not mafioso.is_alive:
            if target:
                game.register_attack(player, target, Attack.BASIC)
                return f"{player.name} is attacking {target.name}."
        return "You have sent the Mafioso to kill."

class Mafioso(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.MAFIOSO
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.attack = Attack.BASIC
        self.is_unique = True
        self.control_immune = True
        self.action_priority = Priority.KILLING
        self.is_roleblock_immune = False
        self.visit_type = VisitType.HARMFUL

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not target:
             return f"{player.name} is not attacking anyone."
        game.register_attack(player, target, self.attack)
        return f"{player.name} is attacking {target.name}."

class SerialKiller(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.SERIAL_KILLER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.attack = Attack.UNSTOPPABLE
        self.defense = Defense.BASIC
        self.cautious = False
        self.is_roleblock_immune = True
        self.detection_immune = True
        self.action_priority = Priority.KILLING
        self.visit_type = VisitType.HARMFUL

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None, cautious: bool = False):
        self.cautious = cautious
        if player.is_role_blocked:
            rb_visitor = next((p for p in player.targeted_by if p.is_alive and p.role.name in [RoleName.TAVERN_KEEPER, RoleName.BOOTLEGGER, RoleName.PIRATE]), None)
            if rb_visitor and not self.cautious:
                game.register_attack(player, rb_visitor, self.attack)
        if target and not player.is_jailed:
            game.register_attack(player, target, self.attack)
            return f"{player.name} is attacking {target.name}."
        return f"{player.name} is staying home."

class Jailor(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.JAILOR
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_unique = True
        self.action_priority = Priority.KILLING
        self.executes = 3
        self.jailed_target = None

    def perform_day_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if target:
            self.jailed_target = target
            target.is_jailed = True
            return f"You have decided to jail {target.name} tonight."
        return None

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not self.jailed_target:
            return "You did not jail anyone."
        if self.executes > 0:
            self.executes -= 1
            self.jailed_target.is_being_executed = True
            game.register_attack(player, self.jailed_target, Attack.UNSTOPPABLE)
            return f"You are executing {self.jailed_target.name}. You have {self.executes} executes left."
        return "You are out of executes."

class Consigliere(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.CONSIGLIERE
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.INVESTIGATION

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not game or not target:
            return "You must select a target."
        return f"Your target, {target.name}, is a {target.role.name.value}."

class Bootlegger(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.BOOTLEGGER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_roleblock_immune = True
        self.action_priority = Priority.CONTROL_PROTECTION

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not target:
            return None
        if target.role.name == RoleName.SERIAL_KILLER and not target.role.cautious:
            return None
        if has_immunity(target.role, ImmunityType.ROLE_BLOCK):
            return f"{target.name} could not be role-blocked."
        target.is_role_blocked = True
        return f"You are distracting {target.name} tonight."

class Arsonist(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.ARSONIST
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.attack = Attack.UNSTOPPABLE
        self.defense = Defense.BASIC
        self.detection_immune = True
        self.action_priority = Priority.SUPPORT_DECEPTION
        # Arsonist only performs an astral (non-visiting) action when igniting; dousing is a regular, non-harmful visit.
        self.visit_type = VisitType.NON_HARMFUL

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not game:
            return "Game state not available."

        # Default to NON_HARMFUL each night; we will override to ASTRAL for ignite/clean actions.
        self.visit_type = VisitType.NON_HARMFUL

        # Cleaning self (no target provided)
        if target is None:
            self.visit_type = VisitType.ASTRAL  # stay home
            if player.is_doused:
                player.is_doused = False
                return "You have cleaned the gasoline off of yourself."
            return "You cleaned yourself of gasoline, though you weren't doused."

        # Ignition (self-target)
        if target == player:
            self.visit_type = VisitType.ASTRAL  # does not visit anyone
            game.ignite_doused_players(player)
            return "You have ignited all doused targets!"

        # Dousing another player
        if not target.is_doused:
            target.is_doused = True
            return f"You have doused {target.name}."
        else:
            return f"{target.name} is already doused."
    
    def perform_day_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None): pass

class Pirate(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.PIRATE
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.attack = Attack.POWERFUL
        self.defense = Defense.NONE
        self.plunders = 0
        self.is_roleblock_immune = True
        self.control_immune = True
        self.detection_immune = True
        self.action_priority = Priority.HIGHEST
        self.duel_target = None
        self.last_dueled_target = None
        self.visit_type = VisitType.HARMFUL

    def perform_day_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if target and target != self.last_dueled_target:
            self.duel_target = target
            return f"{player.name} has decided to duel {target.name} tonight."
        elif target == self.last_dueled_target:
            return "You cannot duel the same player two nights in a row."
        return None

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        actual_target = self.duel_target
        if not actual_target:
            return
        self.last_dueled_target = actual_target
        self.duel_target = None
        if player.is_jailed:
            return "You were hauled off to Jail so you couldn't Duel your target."
        if actual_target.is_jailed:
            return "Your target was hauled off to jail so you couldn't Duel them."
        duel_won, pirate_move, target_defense = self.resolve_duel()
        actual_target.is_role_blocked = True
        if duel_won:
            game.register_attack(player, actual_target, self.attack, is_duel_win=True)
            return f"You chose {pirate_move.value} and defeated {actual_target.name}'s {target_defense.value}. You won the Duel!"
        else:
            if isinstance(actual_target.role, SerialKiller) and actual_target.role.cautious:
                game.register_attack(actual_target, player, actual_target.role.attack, is_primary=False)
            return f"You chose {pirate_move.value} but were bested by {actual_target.name}'s {target_defense.value}. You lost the Duel!"

    def resolve_duel(self):
        pirate_move = random.choice(list(DuelMove))
        target_defense = random.choice(list(DuelDefense))
        win_conditions = {
            (DuelMove.SCIMITAR, DuelDefense.SIDESTEP),
            (DuelMove.RAPIER, DuelDefense.CHAINMAIL),
            (DuelMove.PISTOL, DuelDefense.BACKPEDAL)
        }
        pirate_wins = (pirate_move, target_defense) in win_conditions
        return pirate_wins, pirate_move, target_defense

#--- Placeholder Classes for Roles on the List Without Full Implementation ---

class PlaceholderRole(Role):
    def __init__(self, role_name: RoleName):
        super().__init__()
        self.name = role_name
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None): pass
    def perform_day_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None): pass

class Veteran(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.VETERAN
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_unique = True
        self.attack = Attack.POWERFUL
        self.defense = Defense.BASIC #Becomes INVINCIBLE on alert
        self.is_roleblock_immune = True
        self.control_immune = True
        self.action_priority = Priority.HIGHEST
        self.alerts = 3
        self.is_on_alert = False

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        #The decision to alert is made by targeting self.
        if target == player:
            if self.alerts > 0:
                self.alerts -= 1
                self.is_on_alert = True
                player.defense = Defense.INVINCIBLE #On alert, nothing can kill a vet
                
                #Register an attack against all visitors
                for visitor in player.targeted_by:
                    game.register_attack(player, visitor, self.attack)
                
                return f"You have decided to go on alert. You have {self.alerts} alerts remaining."
            else:
                return "You have no alerts left."
        return "You have decided not to go on alert."

    def perform_day_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        #Reset alert status at the beginning of the day
        self.is_on_alert = False
        #player.defense is reset in player.reset_night_states()

#────────────────────────────────────────────────
#Town Investigative – Psychic
#Astral investigative that alternates Good/Evil visions
class Psychic(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.PSYCHIC
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.visit_type = VisitType.ASTRAL  #does not leave home / cannot be seen
        self.action_priority = Priority.INVESTIGATION

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        """Psychic receives a vision – implemented as a notification added during the night.

        Odd-numbered nights (N1, N3 …) – 3 names, one is Evil
        Even nights – 2 names, one is Good
        We simply sample living players (excluding self) and reveal the list.
        """
        if not game:
            return None

        import random
        candidates = [p for p in game.players if p.is_alive and p != player]
        if len(candidates) < 2:
            return None

        #night index == game.day (day counter increments before night)
        if game.day % 2 == 1:  #Odd night → Evil vision
            sample_size = min(3, len(candidates))
            chosen = random.sample(candidates, sample_size)
            evil = random.choice(chosen)
            msg = f"Your vision: {', '.join(p.name for p in chosen)}. One of them is evil."
            player.notifications.append(msg)
            return "You focus on your crystal ball… a dark presence emerges."
        else:  #Even night → Good vision
            sample_size = min(2, len(candidates))
            chosen = random.sample(candidates, sample_size)
            good = random.choice(chosen)
            msg = f"Your vision: {', '.join(p.name for p in chosen)}. One of them is good."
            player.notifications.append(msg)
            return "Your divination reveals a glimpse of hope."

class Spy(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.SPY
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.FINALIZATION

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        #The spy's main ability is passive and handled in the game loop.
        #This function can be used for "bugging" a target.
        if target:
            #TODO: Implement bugging logic
            return f"You have decided to bug {target.name}'s house."
        return "You are listening for whispers and watching the villains."

class Tracker(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.TRACKER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.INVESTIGATION
        self.visit_type = VisitType.NON_HARMFUL

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not game or not target:
            return "You must select a target."

        #The visits are determined in _process_visits prior to night actions
        visitor_name = target.visiting.name if getattr(target, 'visiting', None) else None
        if visitor_name:
            result = f"Your target visited {visitor_name} tonight."
        else:
            result = "Your target did not visit anyone."
        player.notifications.append(result)
        return result

class Crusader(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.CRUSADER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.CONTROL_PROTECTION

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not target:
            return None
        target.protected_by.append(player)
        return f"You are protecting {target.name} tonight."

class Trapper(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.TRAPPER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.HIGHEST  #Setting trap
        self.has_trap = True

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not self.has_trap:
            return "You are out of traps."
        if not target or not game:
            return "You must select someone to set a trap on."
        game.traps.append({"owner": player, "location": target, "active": True})
        self.has_trap = False
        return f"You have set a trap at {target.name}'s house."

#────────────────────────────────────────────────
#Town Killing – Vampire Hunter
class VampireHunter(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.VAMPIRE_HUNTER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.KILLING
        self.visit_type = VisitType.HARMFUL
        self.attack_vs_vamp = Attack.POWERFUL
        self.attack = Attack.BASIC
        self.shots = None  #unlimited

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not game:
            return None

        #If there are no living vampires, VH becomes a Vigilante automatically (regardless of target)
        living_vamps = [p for p in game.players if p.is_alive and p.role.name == RoleName.VAMPIRE]
        if not living_vamps:
            player.assign_role(Vigilante())
            player.notifications.append("With no vampires left, you have taken up your gun as a Vigilante!")
            return "No vampires remain. You are now a Vigilante."

        # If there are vampires, a target is required to stake
        if not target:
            return "You must choose a target to stake."

        atk = self.attack_vs_vamp if target.role.name == RoleName.VAMPIRE else self.attack
        game.register_attack(player, target, atk)
        return f"You are staking {target.name} tonight."

#────────────────────────────────────────────────
#Town Support – Retributionist
class Retributionist(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.RETRIBUTIONIST
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_unique = True
        self.revived = False  #once-per-game

    def perform_day_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if self.revived or not game or not target:
            return None
        if target.is_alive:
            return "You can only revive dead town members."
        #Only non-unique town roles can be revived (simplified check)
        if target.role.faction != Faction.TOWN or target.role.is_unique:
            return "You cannot revive that role."

        self.revived = True
        target.is_alive = True
        target.defense = target.role.defense
        game.graveyard.remove(target) if target in game.graveyard else None
        target.notifications.append("You have been revived by a Retributionist!")
        return f"You have revived {target.name}."

    def perform_night_action(self, player:'Player', target:'Player'=None, game:'Game'=None):
        return None

class Transporter(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.TRANSPORTER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_roleblock_immune = True
        self.control_immune = True
        self.action_priority = Priority.HIGHEST

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        #The Transporter's action is handled differently in the main game loop
        #to affect other players' targets. This function is a stub.
        return f"You have chosen to transport two people."

#────────────────────────────────────────────────
#Mafia Deception – Disguiser
class Disguiser(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.DISGUISER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.SUPPORT_DECEPTION
        self.visit_type = VisitType.HARMFUL
        self.used = False

    def perform_night_action(self, player:'Player', target:'Player'=None, game:'Game'=None):
        if self.used or not game or not target:
            return None
        #Mark disguise attempt – success if target dies this night handled by game finalization.
        player.disguise_target = target
        self.used = True
        return f"You attempt to disguise yourself as {target.name}."

#Mafia Deception – Forger
class Forger(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.FORGER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.SUPPORT_DECEPTION
        self.visit_type = VisitType.ASTRAL
        self.charges = 2

    def perform_night_action(self, player:'Player', target:'Player'=None, game:'Game'=None):
        if self.charges<=0 or not game or not target:
            return None
        target.forged = True
        self.charges -=1
        return f"You forged {target.name}'s last will. ({self.charges} charges left)"

#Mafia Deception – Framer
class Framer(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.FRAMER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.SUPPORT_DECEPTION
        self.visit_type = VisitType.NON_HARMFUL

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not game or not target:
            return None
        target.is_framed = True
        return f"You framed {target.name} tonight."

#────────────────────────────────────────────────
#Mafia Deception – Hypnotist
class Hypnotist(Role):
    _messages = [
        "You were healed last night!",
        "Someone tried to role block you but failed!",
        "You were transported to another location!",
        "You were visited by a vampire but fought them off!",
        "A bodyguard protected you last night!",
    ]

    def __init__(self):
        super().__init__()
        self.name = RoleName.HYPNOTIST
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.SUPPORT_DECEPTION
        self.visit_type = VisitType.ASTRAL

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not game or not target:
            return None
        import random
        msg = random.choice(self._messages)
        target.notifications.append(msg)
        return f"You hypnotized {target.name} – they will see: '{msg}'"

#Mafia Deception – Janitor
class Janitor(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.JANITOR
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.SUPPORT_DECEPTION
        self.visit_type = VisitType.HARMFUL
        self.charges = 3

    def perform_night_action(self, player:'Player', target:'Player'=None, game:'Game'=None):
        if self.charges<=0 or not game or not target:
            return None
        target.cleaned_by = player
        self.charges -=1
        return f"You cleaned {target.name}'s house. ({self.charges} charges left)"

#────────────────────────────────────────────────
#Neutral Benign – Amnesiac
class Amnesiac(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.AMNESIAC
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.HIGHEST
        self.remembered = False

    def perform_night_action(self, player:'Player', target:'Player'=None, game:'Game'=None):
        if self.remembered or not game or not game.graveyard:
            return None
        #pick first graveyard role or given target
        chosen = game.graveyard[0] if not target else target
        new_role_cls = chosen.role.__class__
        player.assign_role(new_role_cls())
        player.notifications.append(f"You have remembered you are a {chosen.role.name.value}!")
        self.remembered = True
        return f"You remembered being {chosen.role.name.value}."

class Ambusher(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.AMBUSHER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.HIGHEST  #pick ambush location
        self.attack = Attack.BASIC
        self.visit_type = VisitType.HARMFUL
        self.ambush_location: 'Player' | None = None

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not target or not game:
            return "You must select a target to ambush."
        self.ambush_location = target
        return f"You are waiting outside {target.name}'s house to ambush visitors."

class Blackmailer(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.BLACKMAILER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.SUPPORT_DECEPTION

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if target:
            #target.is_blackmailed = True #Cannot do this until player.py is editable
            return f"You have decided to blackmail {target.name}."
        return "You have decided not to blackmail anyone."

    def get_info(self):
        #In a real game, this would allow reading whispers.
        #We can simulate this by giving the Blackmailer access to a whisper channel.
        return {"can_read_whispers": True}

class GuardianAngel(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.GUARDIAN_ANGEL
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.CONTROL_PROTECTION
        self.visit_type = VisitType.ASTRAL
        self.protect_target: 'Player' | None = None
        self.control_immune = True
        self.detection_immune = True

    def assign_protect_target(self, game: 'Game'):
        choices = [p for p in game.players if p != game.get_player_by_name(self.name)]
        if choices:
            self.protect_target = random.choice(choices)

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        #Guardian angel always protects their assigned target, ignoring provided target.
        if not self.protect_target:
            return "No assigned target to protect."
        self.protect_target.protected_by.append(player)
        return f"You are shielding {self.protect_target.name} tonight."

class Survivor(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.SURVIVOR
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.vests = 4
        self.visit_type = VisitType.ASTRAL
        self.action_priority = Priority.SUPPORT_DECEPTION

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if self.vests > 0:
            self.vests -= 1
            player.defense = Defense.BASIC
            return f"You used a bulletproof vest tonight. Vests left: {self.vests}."
        return "You have no vests remaining."

class Executioner(Role):
    """Neutral Evil role whose sole goal is to get their randomly-assigned Town target lynched."""

    def __init__(self):
        super().__init__()
        self.name = RoleName.EXECUTIONER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.FINALIZATION  # very low – no real night action
        self.defense = Defense.BASIC
        self.detection_immune = True
        self.visit_type = VisitType.NON_HARMFUL

        # The Town player the Executioner wants lynched. Chosen at game start.
        self.target: 'Player' | None = None

    def assign_target(self, game: 'Game'):
        """Pick a random Town player other than self. Called by Game at init."""
        town_candidates = [p for p in game.players if p.role.faction == Faction.TOWN and p.name != self.name]
        if town_candidates:
            self.target = random.choice(town_candidates)

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        """Executioner has no night power; return a flavour string for logs."""
        return "You obsess over getting your target lynched…"

class Jester(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.JESTER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_roleblock_immune = False
        self.detection_immune = True
        self.action_priority = Priority.DAY_ACTION #Haunting is a result of a day lynch

    def perform_day_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        #This is for haunting. It should only be called if the jester is lynched.
        if not target:
            return "You must select a target to haunt."
        if not game:
            return
        
        #Jester's haunt is an unstoppable attack
        game.register_attack(player, target, Attack.UNSTOPPABLE, is_haunt=True)
        return f"You have chosen to haunt {target.name}. Their death will be a spectacle!"

    def on_lynch(self, game: 'Game'):
        #In a real game, this would trigger the haunting ability.
        #The player would then choose a target from the players who voted guilty.
        print(f"[Jester] {self.name} was lynched! They will haunt one of their tormentors.")

class Witch(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.WITCH
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.attack = Attack.BASIC
        self.defense = Defense.NONE
        self.is_roleblock_immune = True
        self.detection_immune = True
        self.action_priority = Priority.CONTROL_PROTECTION
        self.has_been_attacked = False

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not game:
            return
        
        try:
            control_target, force_target = game.night_actions[player]
        except (KeyError, TypeError, ValueError):
            return "You must select two targets to perform your dark magic."

        if not control_target or not force_target:
            return "You must select two targets."

        if control_target.role.control_immune:
            control_target.notifications.append("Someone tried to control you but you were immune!")
            return "Your target was immune to control!"

        game.night_actions[control_target] = force_target
        control_target.is_controlled = True
        control_target.notifications.append(f"You were controlled by a Witch! You were forced to target {force_target.name}.")
        
        return f"You have controlled {control_target.name} and forced them to target {force_target.name}."

    def on_attacked(self, attacker):
        if not self.has_been_attacked:
            self.has_been_attacked = True
            print(f"[Witch] The witch was attacked by {attacker.name} and can now curse them.")

#────────────────────────────────────────────────
#Neutral Killing – Juggernaut
class Juggernaut(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.JUGGERNAUT
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.kills = 0
        self.attack = Attack.BASIC
        self.defense = Defense.NONE
        self.action_priority = Priority.KILLING
        self.visit_type = VisitType.HARMFUL

    def _update_power(self):
        if self.kills >= 4:
            self.attack = Attack.UNSTOPPABLE
            self.defense = Defense.INVINCIBLE
        elif self.kills >= 2:
            self.attack = Attack.POWERFUL
            self.defense = Defense.POWERFUL
        elif self.kills >= 1:
            self.attack = Attack.POWERFUL
            self.defense = Defense.BASIC

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not game or not target:
            return None
        self._update_power()
        game.register_attack(player, target, self.attack)
        return f"You are striking {target.name} with {self.attack.name.lower()} force."

    #Called by game when a kill is confirmed (hook not yet wired; simple public method)
    def register_kill(self):
        self.kills += 1
        self._update_power()

#────────────────────────────────────────────────
#Neutral Killing – Werewolf
class Werewolf(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.WEREWOLF
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.KILLING
        self.attack = Attack.UNSTOPPABLE
        self.visit_type = VisitType.HARMFUL  #Rampage at location
        self.full_moon_cycle = 2  #every 2nd night starting N2

    def _is_full_moon(self, game: 'Game'):
        #Night number is game.day; full moon nights are even-numbered nights (N2,4…)
        return game.day % self.full_moon_cycle == 0

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not game or not target:
            return None
        if not self._is_full_moon(game):
            return "It is not a full moon. You stay home snarling."  #WW only acts on full moon

        #Rampage: attack target and all visitors to target
        game.register_attack(player, target, self.attack)
        victims = [v for v in target.targeted_by if v.is_alive]
        for v in victims:
            game.register_attack(player, v, self.attack, is_primary=False)
        return f"You rampage at {target.name}'s house, mauling everyone inside!"

#Neutral Chaos – Plaguebearer / Pestilence
class Plaguebearer(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.PLAGUEBEARER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.infected = set()
        self.action_priority = Priority.SUPPORT_DECEPTION

    def perform_night_action(self, player:'Player', target:'Player'=None, game:'Game'=None):
        if not game:
            return None
        #Infect all players who visit or are visited by PB
        for p in game.players:
            if p.is_alive and (p==player or p in player.targeted_by or p.visiting==player):
                p.is_infected = True
                self.infected.add(p)

        #Check if everyone (non-PB) infected
        others=[p for p in game.players if p.is_alive and p!=player]
        if others and all(p.is_infected for p in others):
            player.assign_role(Pestilence())
            player.notifications.append("You have become Pestilence, Horseman of the Apocalypse!")
        return "You spread disease silently."

class Pestilence(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.PESTILENCE
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.attack = Attack.UNSTOPPABLE
        self.defense = Defense.INVINCIBLE
        self.action_priority = Priority.KILLING
        self.visit_type = VisitType.HARMFUL

    def perform_night_action(self, player:'Player', target:'Player'=None, game:'Game'=None):
        if not game or not target:
            return None
        #Rampage like Werewolf every night
        game.register_attack(player, target, self.attack)
        for v in target.targeted_by:
            if v.is_alive:
                game.register_attack(player, v, self.attack, is_primary=False)
        return f"You rampage at {target.name}."

#Neutral Chaos – Vampire
class Vampire(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.VAMPIRE
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.SUPPORT_DECEPTION
        self.visit_type = VisitType.NON_HARMFUL

    def perform_night_action(self, player:'Player', target:'Player'=None, game:'Game'=None):
        if not game or not target:
            return None
        if target.role.name==RoleName.VAMPIRE or not target.is_alive:
            return "Target unsuitable."
        #Convert target
        target.assign_role(Vampire())
        target.notifications.append("You have been bitten and turned into a Vampire!")
        return f"You bit {target.name}."

class CovenLeader(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.COVEN_LEADER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_coven = True
        self.action_priority = Priority.CONTROL_PROTECTION
        self.is_roleblock_immune = True
        self.control_immune = True
        self.visit_type = VisitType.HARMFUL

    def perform_night_action(self, player:'Player', target:'Player'=None, game:'Game'=None):
        if self.has_necronomicon:
            if target:
                game.register_attack(player, target, Attack.POWERFUL)
                return f"You used the Necronomicon to attack {target.name}."
            return "No target chosen for attack."
        #Without Necronomicon behave like Witch control (simplified one target forces self)
        return "Directing your coven magic (not yet implemented)."

class Medusa(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.MEDUSA
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_coven = True
        self.visit_type = VisitType.ASTRAL
        self.action_priority = Priority.SUPPORT_DECEPTION  #stone gaze before investigator
        self.stone_charges = 3

    def perform_night_action(self, player, target=None, game=None):
        if not game:
            return None

        #With book: can actively visit & attack
        if self.has_necronomicon and target and target != player:
            game.register_attack(player, target, Attack.POWERFUL)
            target.was_stoned = True
            return f"You visited and stoned {target.name}."

        #Without book or self target: use stone gaze on visitors
        if self.stone_charges <= 0:
            return "You are out of stone-gaze charges."
        self.stone_charges -= 1
        victims = [v for v in player.targeted_by if v.is_alive and v != player]
        for v in victims:
            game.register_attack(player, v, Attack.POWERFUL, is_primary=False)
            v.was_stoned = True
        return f"You used stone gaze. Visitors petrified: {', '.join([v.name for v in victims]) if victims else 'none'}. Charges left {self.stone_charges}."

class HexMaster(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.HEX_MASTER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_coven = True
        self.action_priority = Priority.INVESTIGATION
        self.visit_type = VisitType.ASTRAL

    def perform_night_action(self, player, target=None, game=None):
        if not game or not target:
            return "You must choose a target."
        #mark hexed
        target.is_hexed = True
        if self.has_necronomicon:
            game.register_attack(player, target, Attack.BASIC)
            return f"You hexed and attacked {target.name}."
        return f"You hexed {target.name}."

class PotionMaster(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.POTION_MASTER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_coven = True
        self.action_priority = Priority.SUPPORT_DECEPTION

    def perform_night_action(self, player,target=None,game=None):
        if not game or not target:
            return "You chose not to use a potion."
        #Simple rotation of attack / heal / reveal based on night number
        night = game.day
        if night %3==1:
            game.register_attack(player,target,Attack.BASIC if not self.has_necronomicon else Attack.POWERFUL)
            return f"You hurled a harmful potion at {target.name}."
        elif night%3==2:
            target.defense = Defense.BASIC
            target.notifications.append("You feel rejuvenated by alchemical energies!")
            return f"You healed {target.name}."
        else:
            player.notifications.append(f"Your divination reveals {target.name} is {target.role.name.value}.")
            return f"You scry {target.name}'s secrets."

class Poisoner(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.POISONER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_coven = True
        self.action_priority = Priority.KILLING

    def perform_night_action(self, player,target=None,game=None):
        if not game or not target:
            return "You must pick someone to poison."
        #Can't poison same person twice
        if target.is_poisoned:
            return f"{target.name} is already poisoned."
        target.is_poisoned = True
        target.poison_timer = 0
        target.poison_uncurable = self.has_necronomicon
        return f"You have poisoned {target.name}."

class Necromancer(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.NECROMANCER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_coven = True
        self.action_priority = Priority.INVESTIGATION

    def perform_night_action(self, player,target=None,game=None):
        if not game or not target:
            return None
        #Use first coven corpse if none specified
        coven_dead=[p for p in game.graveyard if getattr(p.role,'is_coven',False)]
        if not coven_dead:
            return "No coven corpses to control."
        corpse=coven_dead[0]
        #Basic effect: attack target with Basic strength
        game.register_attack(player,target,Attack.BASIC)
        return f"You animated {corpse.name}'s corpse to attack {target.name}."

#The canonical map of all roles in the game
role_map = {
    RoleName.INVESTIGATOR: Investigator,
    RoleName.LOOKOUT: Lookout,
    RoleName.PSYCHIC: Psychic,
    RoleName.SHERIFF: Sheriff,
    RoleName.SPY: Spy,
    RoleName.TRACKER: Tracker,
    RoleName.BODYGUARD: Bodyguard,
    RoleName.CRUSADER: Crusader,
    RoleName.DOCTOR: Doctor,
    RoleName.TRAPPER: Trapper,
    RoleName.JAILOR: Jailor,
    RoleName.VAMPIRE_HUNTER: VampireHunter,
    RoleName.VETERAN: Veteran,
    RoleName.VIGILANTE: Vigilante,
    RoleName.MAYOR: Mayor,
    RoleName.MEDIUM: Medium,
    RoleName.RETRIBUTIONIST: Retributionist,
    RoleName.TAVERN_KEEPER: TavernKeeper,
    RoleName.TRANSPORTER: Transporter,
    RoleName.DISGUISER: Disguiser,
    RoleName.FORGER: Forger,
    RoleName.FRAMER: Framer,
    RoleName.HYPNOTIST: Hypnotist,
    RoleName.JANITOR: Janitor,
    RoleName.AMBUSHER: Ambusher,
    RoleName.GODFATHER: Godfather,
    RoleName.MAFIOSO: Mafioso,
    RoleName.BLACKMAILER: Blackmailer,
    RoleName.CONSIGLIERE: Consigliere,
    RoleName.AMNESIAC: Amnesiac,
    RoleName.GUARDIAN_ANGEL: GuardianAngel,
    RoleName.SURVIVOR: Survivor,
    RoleName.EXECUTIONER: Executioner,
    RoleName.JESTER: Jester,
    RoleName.WITCH: Witch,
    RoleName.ARSONIST: Arsonist,
    RoleName.PESTILENCE: Pestilence,
    RoleName.PLAGUEBEARER: Plaguebearer,
    RoleName.VAMPIRE: Vampire,
    RoleName.COVEN_LEADER: CovenLeader,
    RoleName.MEDUSA: Medusa,
    RoleName.HEX_MASTER: HexMaster,
    RoleName.POISONER: Poisoner,
    RoleName.POTION_MASTER: PotionMaster,
    RoleName.NECROMANCER: Necromancer,
    #Compatibility for old names, pointing to the new classes
    RoleName.ESCORT: TavernKeeper,
    RoleName.CONSORT: Bootlegger,
    RoleName.SERIAL_KILLER: SerialKiller,
    RoleName.PIRATE: Pirate,
    RoleName.JUGGERNAUT: Juggernaut,
    RoleName.WEREWOLF: Werewolf,
    RoleName.BOOTLEGGER: Bootlegger,
} 
