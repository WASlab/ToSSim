from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import random
from .enums import *
from .config import Perspective, ROLE_ACTION_MESSAGES, CANCEL_ACTION_MESSAGE
from .alignment import get_role_alignment, get_role_faction

if TYPE_CHECKING:
    from .player import Player
    from .game import Game

def get_role_message(role_name: RoleName, action: str, perspective: Perspective, **kwargs) -> str:
    """Helper function to get role action messages using the new dictionary format."""
    try:
        message_template = ROLE_ACTION_MESSAGES[role_name][action][perspective]
        return message_template.format(**kwargs)
    except (KeyError, TypeError):
        # Fallback to a generic message if the specific one isn't found
        return f"Action {action} completed for {role_name.value}"

class Role(ABC):
    def __init__(self):
        self.name: RoleName = None
        self.faction: Faction = None
        self.alignment: RoleAlignment = None
        self.attack: Attack = Attack.NONE
        self.defense: Defense = Defense.NONE
        self.is_unique: bool = False
        self.action_priority = Priority.PRIORITY_3
        self.is_roleblock_immune = False
        self.detection_immune = False
        self.control_immune = False
        #Visit type: default harmful if attacking, else non-harmful
        self.visit_type = VisitType.HARMFUL if self.attack != Attack.NONE else VisitType.NON_HARMFUL
        self.is_coven = False
        self.has_necronomicon = False
        #A message the killer can leave when they successfully kill a victim.
        #This will be displayed the next day alongside the victim's body.
        self.death_note: str = ""
        self.chosen_move = None  # set by InteractionHandler when player specifies weapon

    @abstractmethod
    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        pass

    def perform_day_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        pass

    def get_info(self):
        try:
            name = self.name.value
        except Exception:
            debug_exception(f"Failed to access .value on self.name: {self.name} (type: {type(self.name)})")
            name = str(self.name)
        try:
            faction = self.faction.name
        except Exception:
            debug_exception(f"Failed to access .name on self.faction: {self.faction} (type: {type(self.faction)})")
            faction = str(self.faction)
        try:
            alignment = self.alignment.name
        except Exception:
            debug_exception(f"Failed to access .name on self.alignment: {self.alignment} (type: {type(self.alignment)})")
            alignment = str(self.alignment)
        try:
            attack = self.attack.name
        except Exception:
            debug_exception(f"Failed to access .name on self.attack: {self.attack} (type: {type(self.attack)})")
            attack = str(self.attack)
        try:
            defense = self.defense.name
        except Exception:
            debug_exception(f"Failed to access .name on self.defense: {self.defense} (type: {type(self.defense)})")
            defense = str(self.defense)
        return {
            "name": name,
            "faction": faction,
            "alignment": alignment,
            "attack": attack,
            "defense": defense,
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
        self.action_priority = Priority.PRIORITY_4

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game'):
        # Handle Disguiser effect: examine displayed role
        eff_role = target.role
        if getattr(target, 'disguised_as_role', None):
            eff_role = create_role_from_name(target.disguised_as_role)

        # Being framed or hexed makes a Townie appear suspicious.
        if target.is_framed or target.is_hexed:
            return get_role_message(self.name, "interrogate", Perspective.SELF, target=target.name)
        if eff_role.detection_immune:
            return get_role_message(self.name, "interrogate", Perspective.SELF, target=target.name)
        if eff_role.name == RoleName.SERIAL_KILLER:
            return get_role_message(self.name, "interrogate", Perspective.SELF, target=target.name)
        if eff_role.faction == Faction.MAFIA and eff_role.name != RoleName.GODFATHER:
            return get_role_message(self.name, "interrogate", Perspective.SELF, target=target.name)
        return get_role_message(self.name, "interrogate", Perspective.SELF, target=target.name)

class Investigator(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.INVESTIGATOR
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.PRIORITY_4

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game'):
        if not game or not target:
            return CANCEL_ACTION_MESSAGE
        investigated_role = target.disguised_as_role if getattr(target,'disguised_as_role',None) else target.role.name
        if target.is_framed or (hasattr(game, 'config') and game.config.is_coven and target.is_hexed):
            investigated_role = RoleName.HEX_MASTER if game.config.is_coven else RoleName.FRAMER
        elif target.is_doused:
            investigated_role = RoleName.ARSONIST
        
        if hasattr(game, 'config'):
            result_group = game.config.get_investigator_result_group(investigated_role)
            if not result_group:
                return get_role_message(self.name, "investigate", Perspective.SELF, target=target.name)
            role_names = [role.value for role in result_group]
            if len(role_names) == 1:
                return get_role_message(self.name, "investigate", Perspective.SELF, target=target.name)
            result_string = ", ".join(role_names[:-1]) + ", or " + role_names[-1]
            return get_role_message(self.name, "investigate", Perspective.SELF, target=target.name)
        
        #Fallback if no config is present
        return get_role_message(self.name, "investigate", Perspective.SELF, target=target.name)

class Lookout(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.LOOKOUT
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.PRIORITY_4

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game'):
        from .enums import VisitType
        visitor_names = []
        
        for visitor in target.targeted_by:
            if visitor.role.visit_type == VisitType.ASTRAL:
                continue  # Astral visits are invisible to Lookout
                
            # Handle Disguiser effect: show disguised role name instead of actual visitor
            visitor_name = visitor.name
            if hasattr(visitor, 'disguised_as_role') and visitor.disguised_as_role:
                # Find the player whose role the visitor is disguised as
                disguise_source = None
                for p in game.players:
                    if p.role.name == visitor.disguised_as_role:
                        disguise_source = p
                        break
                if disguise_source:
                    visitor_name = disguise_source.name
                    
            visitor_names.append(visitor_name)
            
        if not visitor_names:
            return get_role_message(self.name, "watch", Perspective.SELF, target=target.name)
        return get_role_message(self.name, "watch", Perspective.SELF, target=target.name)

class Doctor(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.DOCTOR
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.PRIORITY_3
        self.self_heals = 1

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not target:
            return CANCEL_ACTION_MESSAGE
        if hasattr(target.role, 'revealed') and getattr(target.role, 'revealed', False):
            return get_role_message(self.name, "heal", Perspective.SELF, target=target.name)
        if player == target:
            if self.self_heals > 0:
                self.self_heals -= 1
                target.defense = Defense.POWERFUL
                target._was_healed_tonight = True  # Mark for notification
                return get_role_message(self.name, "self_heal", Perspective.SELF, target=target.name)
            else:
                return get_role_message(self.name, "self_heal", Perspective.SELF, target=target.name)
        target.defense = Defense.POWERFUL
        target._was_healed_tonight = True  # Mark for notification
        return get_role_message(self.name, "heal_target", Perspective.SELF, target_name=target.name)

class Bodyguard(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.BODYGUARD
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.PRIORITY_3
        self.vests = 1

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game'):
        if not target:
            return CANCEL_ACTION_MESSAGE
        if player == target:
            if self.vests > 0:
                self.vests -= 1
                player.defense = Defense.BASIC
                return get_role_message(self.name, "vest", Perspective.SELF, target=target.name)
            else:
                return get_role_message(self.name, "vest", Perspective.SELF, target=target.name)
        if target:
            target.protected_by.append(player)
            return get_role_message(self.name, "guard", Perspective.SELF, target=target.name)
        return None

class Vigilante(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.VIGILANTE
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.attack = Attack.BASIC
        self.action_priority = Priority.PRIORITY_5
        self.bullets = 3
        self.has_killed_townie = False
        self.put_gun_down = False
        self.visit_type = VisitType.HARMFUL

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not target or not game:
            return CANCEL_ACTION_MESSAGE
        
        if self.bullets > 0:
            self.bullets -= 1
            game.register_attack(player, target, self.attack)
            return get_role_message(self.name, "shoot_target", Perspective.SELF, target_name=target.name, bullets_left=self.bullets)
        else:
            return get_role_message(self.name, "shoot_no_bullets", Perspective.SELF)

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
            player.vote_weight = 3
            # The notification is now handled by the game engine when the action result is returned.
            print(f"{player.name} has revealed themselves as the Mayor!")
            return True
        return False
    
    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        return None

    def perform_day_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        #Mayor can click a reveal button during the day; no target needed.
        if not self.revealed:
            success = self.reveal(player)
            if success:
                #Update player's vote weight
                player.vote_weight = 3
                return get_role_message(self.name, "reveal_success", Perspective.SELF)
        return None

class Medium(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.MEDIUM
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.PRIORITY_1
        self.seances = 1
        self.can_seance_from_grave = True

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not target:
            return CANCEL_ACTION_MESSAGE
        if self.seances > 0:
            return self.seance(player, target, game)
        return get_role_message(self.name, "speaking_with_dead", Perspective.SELF)
    
    def seance(self, medium_player: 'Player', target: 'Player', game: 'Game'):
        """Create a private seance channel between Medium and target."""
        if self.seances <= 0:
            return get_role_message(self.name, "seance_no_charges", Perspective.SELF)
        
        if not target or not target.is_alive:
            return get_role_message(self.name, "seance_target_not_alive", Perspective.SELF)
        
        if medium_player == target:
            return get_role_message(self.name, "seance_self_target", Perspective.SELF)
        
        # Use the seance
        self.seances -= 1
        
        # Create the seance channel through the chat system
        if hasattr(game, 'chat'):
            game.chat.create_seance_channel(medium_player, target)
            return get_role_message(self.name, "seance_success", Perspective.SELF, target_name=target.name)
        
        return get_role_message(self.name, "seance_attempt", Perspective.SELF, target_name=target.name)

class TavernKeeper(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.TAVERN_KEEPER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_roleblock_immune = True
        self.action_priority = Priority.PRIORITY_2_HIGHEST

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not target:
            return CANCEL_ACTION_MESSAGE
        if target.role.name == RoleName.SERIAL_KILLER and not target.role.cautious:
            return None # Tavern Keeper is killed by uncautious SK
        if has_immunity(target.role, ImmunityType.ROLE_BLOCK):
            # The target will receive their own notification about being immune.
            return get_role_message(self.name, "roleblock_immune", Perspective.SELF, target_name=target.name)
        target.is_role_blocked = True
        return get_role_message(self.name, "distract_target", Perspective.SELF, target_name=target.name)

class Godfather(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.GODFATHER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.defense = Defense.BASIC
        self.is_unique = True
        self.detection_immune = True
        self.action_priority = Priority.PRIORITY_5
        self.is_roleblock_immune = False 
        self.visit_type = VisitType.HARMFUL

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not target:
            return CANCEL_ACTION_MESSAGE
        mafioso = game.find_player_by_role(RoleName.MAFIOSO)
        if not mafioso or not mafioso.is_alive:
            # If no Mafioso, Godfather attacks directly
            game.register_attack(player, target, Attack.BASIC)
            return get_role_message(self.name, "attack_target", Perspective.SELF, target_name=target.name)
        return get_role_message(self.name, "send_mafioso_to_kill", Perspective.SELF)

class Mafioso(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.MAFIOSO
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.attack = Attack.BASIC
        self.is_unique = True
        self.control_immune = True
        self.action_priority = Priority.PRIORITY_5
        self.is_roleblock_immune = False
        self.visit_type = VisitType.HARMFUL

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not target:
             return CANCEL_ACTION_MESSAGE
        game.register_attack(player, target, self.attack)
        return get_role_message(self.name, "attack_target", Perspective.SELF, target_name=target.name)

class SerialKiller(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.SERIAL_KILLER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        # Per official Town of Salem rules, Serial Killer has BASIC attack (not unstoppable)
        self.attack = Attack.BASIC
        self.defense = Defense.BASIC
        self.cautious = False
        self.is_roleblock_immune = True
        self.detection_immune = True
        self.action_priority = Priority.PRIORITY_5
        self.visit_type = VisitType.HARMFUL

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None, cautious: bool = False):
        self.cautious = cautious
        
        # Check if any role blockers are visiting
        role_blocker_visitors = [p for p in player.targeted_by if p.is_alive and p.role.name in [RoleName.TAVERN_KEEPER, RoleName.BOOTLEGGER, RoleName.PIRATE]]
        
        if role_blocker_visitors:
            if self.cautious:
                # Cautious SK refuses to attack if role blockers visit (stays home to avoid detection)
                return get_role_message(self.name, "stay_home_cautious", Perspective.SELF)
            else:
                # Normal SK attacks the role blocker(s) who visit
                for rb_visitor in role_blocker_visitors:
                    game.register_attack(player, rb_visitor, self.attack)
                    rb_visitor.last_will_bloodied = True
                # Normal SK also attacks their intended target if not jailed
                if target and not player.is_jailed:
                    game.register_attack(player, target, self.attack)
                    return get_role_message(self.name, "attack_target_and_visitors", Perspective.SELF, target_name=target.name)
                else:
                    return get_role_message(self.name, "attack_visitors_only", Perspective.SELF)
        
        # No role blockers visiting - proceed with normal attack
        if target and not player.is_jailed:
            game.register_attack(player, target, self.attack)
            return get_role_message(self.name, "attack_target", Perspective.SELF, target_name=target.name)
        return get_role_message(self.name, "stay_home", Perspective.SELF)

class Jailor(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.JAILOR
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_unique = True
        self.action_priority = Priority.PRIORITY_5
        self.executes = 3
        self.jailed_target = None
        self.is_roleblock_immune = False  # Jailor can be roleblocked

    def perform_day_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not target:
            return CANCEL_ACTION_MESSAGE
        self.jailed_target = target
        target.is_jailed = True
        return get_role_message(self.name, "jail_target", Perspective.SELF, target_name=target.name)

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if game.day == 0:
            return get_role_message(self.name, "execute_day_zero", Perspective.SELF)
        if not self.jailed_target:
            return get_role_message(self.name, "execute_no_target", Perspective.SELF)
        
        # Check if Jailor is role-blocked
        if player.is_role_blocked:
            return get_role_message(self.name, "execute_roleblocked", Perspective.SELF)
            
        if self.executes > 0:
            # Check if target cannot be executed
            if self.jailed_target.role.name == RoleName.PESTILENCE:
                return get_role_message(self.name, "execute_pestilence_immune", Perspective.SELF)
            
            # Check if Guardian Angel is protecting the target
            if hasattr(self.jailed_target, 'protected_by') and self.jailed_target.protected_by:
                for protector in self.jailed_target.protected_by:
                    if protector.role.name == RoleName.GUARDIAN_ANGEL:
                        return get_role_message(self.name, "execute_ga_protected", Perspective.SELF)
            
            self.executes -= 1
            self.jailed_target.is_being_executed = True
            # TODO: RESEARCH METRICS - Track execution events
            self.jailed_target.research_metrics['times_executed'] += 1
            self.jailed_target.research_metrics['death_cause'] = 'executed'
            game.register_attack(player, self.jailed_target, Attack.UNSTOPPABLE)
            return get_role_message(self.name, "execute_success", Perspective.SELF, target_name=self.jailed_target.name, executes_left=self.executes)
        return get_role_message(self.name, "execute_no_charges", Perspective.SELF)

class Consigliere(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.CONSIGLIERE
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.PRIORITY_4

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not game or not target:
            return CANCEL_ACTION_MESSAGE
        return game.config.get_consigliere_result(target.role.name)

class Bootlegger(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.BOOTLEGGER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        # Bootlegger has roleblock immunity like Escort/Consort
        self.is_roleblock_immune = True
        self.action_priority = Priority.PRIORITY_2_HIGHEST

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not target:
            return CANCEL_ACTION_MESSAGE
        if target.role.name == RoleName.SERIAL_KILLER and not target.role.cautious:
            return None # Bootlegger is killed by uncautious SK
        if has_immunity(target.role, ImmunityType.ROLE_BLOCK):
            return get_role_message(self.name, "roleblock_immune", Perspective.SELF, target_name=target.name)
        target.is_role_blocked = True
        return get_role_message(self.name, "distract_target", Perspective.SELF, target_name=target.name)

class Arsonist(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.ARSONIST
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.attack = Attack.UNSTOPPABLE
        self.defense = Defense.BASIC
        self.detection_immune = True
        self.action_priority = Priority.PRIORITY_3  # Dousing is Priority 3, igniting is Priority 5
        # Arsonist only performs an astral (non-visiting) action when igniting; dousing is a regular, non-harmful visit.
        self.visit_type = VisitType.NON_HARMFUL

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not game:
            return CANCEL_ACTION_MESSAGE.get_message(self.name, Perspective.SELF, "game_state_unavailable")

        # Default to NON_HARMFUL each night; we will override to ASTRAL for ignite/clean actions.
        self.visit_type = VisitType.NON_HARMFUL

        # Cleaning self (no target provided)
        if target is None:
            self.visit_type = VisitType.ASTRAL  # stay home
            if player.is_doused:
                player.is_doused = False
                return get_role_message(self.name, "clean_self_doused", Perspective.SELF)
            return get_role_message(self.name, "clean_self_not_doused", Perspective.SELF)

        # Ignition (self-target)
        if target == player:
            self.visit_type = VisitType.ASTRAL  # does not visit anyone
            game.ignite_doused_players(player)
            return get_role_message(self.name, "ignite_success", Perspective.SELF)

        # Dousing another player
        if not target.is_doused:
            target.is_doused = True
            return get_role_message(self.name, "douse_target", Perspective.SELF, target_name=target.name)
        else:
            return get_role_message(self.name, "target_already_doused", Perspective.SELF, target_name=target.name)
    
    def perform_day_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None): pass

class Pirate(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.PIRATE
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.attack = Attack.POWERFUL
        self.defense = Defense.NONE
        self.plunders = 0  # Need 2 successful plunders to win
        self.is_roleblock_immune = True
        self.control_immune = True
        self.detection_immune = True
        self.action_priority = Priority.PRIORITY_1
        self.duel_target = None
        self.last_dueled_target = None
        self.chosen_move = None  # Player's chosen duel move
        self.visit_type = VisitType.HARMFUL

    def perform_day_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not target:
            return CANCEL_ACTION_MESSAGE
        if target == self.last_dueled_target:
            return get_role_message(self.name, "duel_same_target_twice", Perspective.SELF)
        self.duel_target = target
        return get_role_message(self.name, "duel_target_chosen", Perspective.SELF, target_name=target.name)

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        actual_target = self.duel_target
        if not actual_target:
            return get_role_message(self.name, "no_duel_target", Perspective.SELF)
            
        self.last_dueled_target = actual_target
        self.duel_target = None
        
        # Check if either player is jailed
        if player.is_jailed:
            return get_role_message(self.name, "duel_jailed_self", Perspective.SELF)
        if actual_target.is_jailed:
            return get_role_message(self.name, "duel_jailed_target", Perspective.SELF)
            
        # Resolve the duel
        duel_won, pirate_move, target_defense = self.resolve_duel()
        
        # Target is always role-blocked by the duel (win or lose)
        actual_target.is_role_blocked = True
        
        if duel_won:
            # Pirate wins - kill the target (plunder will be awarded by game engine)
            game.register_attack(player, actual_target, self.attack, is_duel_win=True)
                
            return get_role_message(self.name, "duel_win", Perspective.SELF, pirate_move=pirate_move.value, target_defense=target_defense.value, plunders_left=self.plunders + 1)
        else:
            # Pirate loses - check for Serial Killer cautious interaction
            if isinstance(actual_target.role, SerialKiller) and not actual_target.role.cautious:
                # Non-cautious SK kills the Pirate for visiting
                game.register_attack(actual_target, player, actual_target.role.attack, is_primary=False)
                player.last_will_bloodied = True
                return get_role_message(self.name, "duel_lose_sk_killed", Perspective.SELF, pirate_move=pirate_move.value, target_defense=target_defense.value)
            
            return get_role_message(self.name, "duel_lose", Perspective.SELF, pirate_move=pirate_move.value, target_defense=target_defense.value)

    def resolve_duel(self):
        """Resolve the pirate duel using rock-paper-scissors style mechanics.
        
        Pirate wins if:
        - Scimitar beats Sidestep (slashing beats dodging)
        - Rapier beats Chainmail (piercing beats armor) 
        - Pistol beats Backpedal (ranged beats retreat)
        """
        import random
        from .enums import DuelMove, DuelDefense
        
        # Pirate chooses move (random if not specified by player)
        pirate_move = self.chosen_move if self.chosen_move else random.choice(list(DuelMove))
        
        # Target randomly chooses defense
        target_defense = random.choice(list(DuelDefense))
        
        # Define win conditions (pirate perspective)
        win_conditions = {
            (DuelMove.SCIMITAR, DuelDefense.SIDESTEP),
            (DuelMove.RAPIER, DuelDefense.CHAINMAIL),
            (DuelMove.PISTOL, DuelDefense.BACKPEDAL)
        }
        
        pirate_wins = (pirate_move, target_defense) in win_conditions
        
        # Reset chosen move after use so next duel defaults to random unless specified
        self.chosen_move = None
        
        return pirate_wins, pirate_move, target_defense

#--- Placeholder Classes for Roles on the List Without Full Implementation ---

class PlaceholderRole(Role):
    def __init__(self, role_name: RoleName):
        super().__init__()
        # Ensure self.name is always a RoleName enum member
        if isinstance(role_name, RoleName):
            self.name = role_name
        else:
            try:
                self.name = RoleName(role_name)
            except ValueError:
                # Fallback if string cannot be converted to RoleName enum
                self.name = RoleName.INVESTIGATOR # Assign a safe default
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
        # In ToS-1 a Veteran has NO defense by default and gains BASIC defense only on alert.
        self.defense = Defense.NONE
        self.is_roleblock_immune = True
        self.control_immune = True
        self.action_priority = Priority.PRIORITY_1_HIGHEST
        self.alerts = 3
        self.is_on_alert = False

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        #The decision to alert is made by targeting self.
        if not target:
            return CANCEL_ACTION_MESSAGE
        if target == player:
            if self.alerts > 0:
                self.alerts -= 1
                self.is_on_alert = True
                player.defense = Defense.BASIC  # On alert the Vet gains BASIC defense
                return get_role_message(self.name, "alert_success", Perspective.SELF, alerts_left=self.alerts)
            else:
                return get_role_message(self.name, "alert_no_charges", Perspective.SELF)
        return get_role_message(self.name, "no_alert_chosen", Perspective.SELF)

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
        self.action_priority = Priority.PRIORITY_4

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        """Psychic receives a vision – implemented as a notification added during the night.

        Odd-numbered nights (N1, N3 …) – 3 names, one is Evil
        Even nights – 2 names, one is Good
        We simply sample living players (excluding self) and reveal the list.
        """
        if not game:
            return CANCEL_ACTION_MESSAGE.get_message(self.name, Perspective.SELF, "game_state_unavailable")

        import random
        candidates = [p for p in game.players if p.is_alive and p != player]
        if len(candidates) < 2:
            return get_role_message(self.name, "psychic_not_enough_players", Perspective.SELF)

        #night index == game.day (day counter increments before night)
        if game.day % 2 == 1:  #Odd night → Evil vision
            sample_size = min(3, len(candidates))
            chosen = random.sample(candidates, sample_size)
            evil = random.choice(chosen)
            msg = get_role_message(self.name, "psychic_evil_vision", Perspective.SELF, player_list=', '.join(p.name for p in chosen))
            game.chat.add_player_notification(player, msg)
            return get_role_message(self.name, "psychic_focus_evil", Perspective.SELF)
        else:  #Even night → Good vision
            sample_size = min(2, len(candidates))
            chosen = random.sample(candidates, sample_size)
            good = random.choice(chosen)
            msg = get_role_message(self.name, "psychic_good_vision", Perspective.SELF, player_list=', '.join(p.name for p in chosen))
            game.chat.add_player_notification(player, msg)
            return get_role_message(self.name, "psychic_focus_good", Perspective.SELF)

class Spy(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.SPY
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.PRIORITY_6

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        # The Spy's passive intel is processed in Game._process_spy_intel.
        # Here we simply confirm the bug choice for player feedback.
        if not target:
            return CANCEL_ACTION_MESSAGE
        return get_role_message(self.name, "bug_target", Perspective.SELF, target_name=target.name)

class Tracker(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.TRACKER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.PRIORITY_3
        self.visit_type = VisitType.NON_HARMFUL

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not game or not target:
            return CANCEL_ACTION_MESSAGE
        
        # Get all non-astral visits made by the target
        from .enums import VisitType
        non_astral_visits = []
        
        # Check each visit to see if it should be visible to Tracker
        for visit_target in getattr(target, 'visiting_all', []):
            # Find the role's visit type by looking up the night action
            if target in game.night_actions:
                # Tracker can see all visits except astral ones
                if target.role.visit_type != VisitType.ASTRAL:
                    non_astral_visits.append(visit_target)
        
        if non_astral_visits:
            if len(non_astral_visits) == 1:
                result = get_role_message(self.name, "track_single_visit", Perspective.SELF, target_name=target.name, visited_player=non_astral_visits[0].name)
            else:
                names = [v.name for v in non_astral_visits]
                result = get_role_message(self.name, "track_multiple_visits", Perspective.SELF, target_name=target.name, visited_players=', '.join(names))
        else:
            result = get_role_message(self.name, "track_no_visits", Perspective.SELF, target_name=target.name)
        
        game.chat.add_player_notification(player, result)
        return result

class Crusader(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.CRUSADER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.PRIORITY_3

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not target:
            return CANCEL_ACTION_MESSAGE
        target.protected_by.append(player)
        return get_role_message(self.name, "protect_target", Perspective.SELF, target_name=target.name)

class Trapper(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.TRAPPER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.PRIORITY_1
        # Internal state
        self.building = False        # trap under construction (will arm next night)
        self.active = False          # trap is armed and added to game.traps
        self.trap_location = None    # Player where trap is/will be located

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        """Handles building, arming, and dismantling a trap.

        • Choose another player ⇒ begin building a trap there (takes one night).
        • Choose self ⇒ dismantle current (building or active) trap.
        """
        if not game:
            return CANCEL_ACTION_MESSAGE.get_message(self.name, Perspective.SELF, "game_state_unavailable")

        # Arm trap if construction finished (this happens at start of the *second* night)
        if self.building and not self.active:
            self.active = True
            self.building = False
            game.traps.append({"owner": player, "location": self.trap_location, "active": True})

        # --- Handle tonight's order ---
        if target == player:  # Attempt to dismantle
            if self.building:
                self.building = False
                self.trap_location = None
                return get_role_message(self.name, "dismantle_building_trap", Perspective.SELF)
            if self.active:
                # deactivate trap in game.traps
                for tr in game.traps:
                    if tr["owner"] == player and tr["active"]:
                        tr["active"] = False
                        break
                self.active = False
                self.trap_location = None
                return get_role_message(self.name, "dismantle_active_trap", Perspective.SELF)
            return get_role_message(self.name, "no_trap_to_dismantle", Perspective.SELF)

        # Building new trap
        if self.building or self.active:
            return get_role_message(self.name, "trap_already_exists", Perspective.SELF)

        if not target:
            return CANCEL_ACTION_MESSAGE

        self.building = True
        self.trap_location = target
        return get_role_message(self.name, "build_trap", Perspective.SELF, target_name=target.name)

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
        self.defense = Defense.BASIC  # Basic defense against Vampires
        self.detection_immune = True

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not game:
            return CANCEL_ACTION_MESSAGE.get_message(self.name, Perspective.SELF, "game_state_unavailable")

        # Auto-promote to Vigilante once all Vampires are dead.
        if not any(p.is_alive and p.role.name == RoleName.VAMPIRE for p in game.players):
            if self.name != RoleName.VIGILANTE:  # Prevent multiple promotions
                new_role = Vigilante()
                new_role.bullets = 1
                player.assign_role(new_role)
                game.chat.add_player_notification(player, get_role_message(self.name, "promote_to_vigilante", Perspective.SELF))
            return get_role_message(self.name, "already_vigilante", Perspective.SELF)  # No further action tonight.

        # If no target supplied just wait / listen to chat.
        if not target:
            return get_role_message(self.name, "no_target_listen_for_undead", Perspective.SELF)

        if target.role.name == RoleName.VAMPIRE:
            # Stake the Vampire with Powerful attack
            game.register_attack(player, target, Attack.POWERFUL)
            return get_role_message(self.name, "stake_vampire", Perspective.SELF, target_name=target.name)
        else:
            return get_role_message(self.name, "inspect_not_vampire", Perspective.SELF, target_name=target.name)

#────────────────────────────────────────────────
#Town Support – Raises a dead Town corpse each night to use its ability once.
#
# Mechanics implemented per ToS 1 / ToS 2 wiki:
#   • Unique role (only one at a time).
#   • Unlimited uses; each corpse can only be used once (rots afterwards).
#   • Role-block & Control immune.
#   • Cannot raise certain roles (Psychic, Trapper, Transporter, Mayor, Medium, Jailor, Veteran, another Retributionist, unique roles).
class Retributionist(Role):
    """Town Support – Raises a dead Town corpse each night to use its ability once.

    Mechanics implemented per ToS 1 / ToS 2 wiki:
      • Unique role (only one at a time).
      • Unlimited uses; each corpse can only be used once (rots afterwards).
      • Role-block & Control immune.
      • Cannot raise certain roles (Psychic, Trapper, Transporter, Mayor, Medium, Jailor, Veteran, another Retributionist, unique roles).
    """

    _UNUSABLE_ROLES = {
        RoleName.PSYCHIC,
        RoleName.TRAPPER,
        RoleName.TRANSPORTER,
        RoleName.MAYOR,
        RoleName.MEDIUM,
        RoleName.JAILOR,
        RoleName.VETERAN,
        RoleName.RETRIBUTIONIST,
    }

    def __init__(self):
        super().__init__()
        self.name = RoleName.RETRIBUTIONIST
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_unique = True
        self.is_roleblock_immune = True
        self.control_immune = True
        # Track corpses already used so they "rot" and cannot be reused.
        self._used_corpses: set['Player'] = set()

    # Retributionist no longer has a day action; all logic happens at night.
    def perform_day_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        return None

    def perform_night_action(self, player: 'Player', target: 'tuple[Player, Player]' = None, game: 'Game' = None):
        """Expect *target* to be a tuple (corpse, second_target)."""
        if not game or not target or not isinstance(target, tuple) or len(target) != 2:
            return CANCEL_ACTION_MESSAGE.get_message(self.name, Perspective.SELF, "invalid_target_format")

        corpse, second_target = target

        # Validate corpse eligibility
        if corpse.is_alive:
            return get_role_message(self.name, "raise_corpse_alive", Perspective.SELF, corpse_name=corpse.name)
        if corpse in self._used_corpses:
            return get_role_message(self.name, "raise_corpse_rotted", Perspective.SELF, corpse_name=corpse.name)
        if corpse.role.faction != Faction.TOWN:
            return get_role_message(self.name, "raise_corpse_not_town", Perspective.SELF, corpse_name=corpse.name)
        if corpse.role.is_unique or corpse.role.name in self._UNUSABLE_ROLES:
            return get_role_message(self.name, "raise_corpse_unusable_role", Perspective.SELF, corpse_role=corpse.role.name.value)

        # Temporarily re-animate the corpse to perform its ability.
        original_state = corpse.is_alive
        corpse.is_alive = True  # allow ability code to run
        
        # Add notification to the corpse about being raised
        game.chat.add_player_notification(corpse, get_role_message(self.name, "corpse_raised", Perspective.OTHER))

        try:
            result = corpse.role.perform_night_action(corpse, second_target, game)
        finally:
            # Ensure corpse returns to dead state regardless of errors.
            corpse.is_alive = original_state

        # Mark corpse as used so it "rots".
        self._used_corpses.add(corpse)

        # Tracker/Lookout visibility (simplified): record visits for corpse
        corpse.visit(second_target)

        return get_role_message(self.name, "raise_success", Perspective.SELF, corpse_name=corpse.name, result=result)

class Transporter(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.TRANSPORTER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_roleblock_immune = True
        self.control_immune = True
        self.action_priority = Priority.PRIORITY_1_HIGHEST
        self.visit_type = VisitType.NON_HARMFUL
        self.transport_targets = None  # Tuple of (player1, player2) to transport

    def perform_night_action(self, player: 'Player', target: 'tuple[Player, Player]' = None, game: 'Game' = None):
        if not game or not target or not isinstance(target, tuple) or len(target) != 2:
            return CANCEL_ACTION_MESSAGE.get_message(self.name, Perspective.SELF, "invalid_target_format")

        a, b = target
        
        # Cannot transport same person with themselves
        if a == b:
            return get_role_message(self.name, "transport_self_target", Perspective.SELF)
        
        # Cannot transport jailed players
        if a.is_jailed or b.is_jailed:
            return get_role_message(self.name, "transport_jailed_target", Perspective.SELF)
        
        # Store transport targets for processing in game
        self.transport_targets = (a, b)
        
        # Visit both targets
        player.visit(a)
        player.visit(b)
        
        return get_role_message(self.name, "transport_success", Perspective.SELF, target1_name=a.name, target2_name=b.name)

#────────────────────────────────────────────────
#Mafia Deception – Disguiser
class Disguiser(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.DISGUISER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.PRIORITY_3
        self.visit_type = VisitType.NON_HARMFUL  # Disguiser doesn't trigger BG/Trap
        self.disguise_target = None  # Target to disguise as
        self.mafia_target = None     # Mafia member to disguise

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not game or not target:
            return CANCEL_ACTION_MESSAGE

        if target.role.faction == Faction.MAFIA:
            return get_role_message(self.name, "disguise_target_is_mafia", Perspective.SELF)
        
        if target.is_jailed:
            return get_role_message(self.name, "disguise_target_jailed", Perspective.SELF)

        # Find a living Mafia member to disguise (prioritize non-Disguiser)
        mafia_members = [p for p in game.players if p.is_alive and p.role.faction == Faction.MAFIA and p != player]
        if not mafia_members:
            return get_role_message(self.name, "no_mafia_to_disguise", Perspective.SELF)
        
        # Choose the first available Mafia member (could be randomized)
        mafia_member = mafia_members[0]
        
        # Set up the disguise - it will take effect when investigations happen
        # Ensure disguised_role_name is always a RoleName enum member
        disguised_role_name = target.role.name
        if not isinstance(disguised_role_name, RoleName):
            try:
                disguised_role_name = RoleName(disguised_role_name)
            except ValueError:
                # Fallback if string cannot be converted to RoleName enum
                # This should ideally not happen if target.role.name is always a RoleName enum
                disguised_role_name = RoleName.INVESTIGATOR # Assign a safe default
        mafia_member.disguised_as_role = disguised_role_name
        self.disguise_target = target
        self.mafia_target = mafia_member
        
        return get_role_message(self.name, "disguise_success", Perspective.SELF, mafia_member_name=mafia_member.name, target_role=target.role.name.value)

#Mafia Deception – Forger
class Forger(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.FORGER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.PRIORITY_3
        self.visit_type = VisitType.ASTRAL
        self.charges = 2

    def perform_night_action(self, player:'Player', target:'Player'=None, game:'Game'=None):
        if self.charges <= 0:
            return get_role_message(self.name, "forger_no_charges", Perspective.SELF)
        if not game or not target:
            return CANCEL_ACTION_MESSAGE

        target.was_forged = True
        self.charges -= 1
        return get_role_message(self.name, "forger_success", Perspective.SELF, target_name=target.name, charges_left=self.charges)

#Mafia Deception – Framer
class Framer(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.FRAMER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.PRIORITY_3
        self.visit_type = VisitType.NON_HARMFUL

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not game or not target:
            return CANCEL_ACTION_MESSAGE
        target.is_framed = True
        return get_role_message(self.name, "frame_target", Perspective.SELF, target_name=target.name)

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
        self.action_priority = Priority.PRIORITY_3
        self.visit_type = VisitType.NON_HARMFUL

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not game or not target:
            return CANCEL_ACTION_MESSAGE
        import random
        msg = random.choice(self._messages)
        game.chat.add_player_notification(target, msg)
        return get_role_message(self.name, "hypnotize_success", Perspective.SELF, target_name=target.name, hypnotized_message=msg)

#Mafia Deception – Janitor
class Janitor(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.JANITOR
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.PRIORITY_3
        self.visit_type = VisitType.HARMFUL
        self.charges = 3

    def perform_night_action(self, player:'Player', target:'Player'=None, game:'Game'=None):
        if not game or not target:
            return CANCEL_ACTION_MESSAGE

        if self.charges <= 0:
            return get_role_message(self.name, "janitor_no_charges", Perspective.SELF)

        # Mark the target for potential cleaning if they die.
        target.cleaned_by = player

        # We will only consume a charge if the clean actually triggers (handled in Game._announce_deaths).
        return get_role_message(self.name, "janitor_attempt_clean", Perspective.SELF, target_name=target.name)

#────────────────────────────────────────────────
#Neutral Benign – Amnesiac
class Amnesiac(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.AMNESIAC
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.detection_immune = True
        self.visit_type = VisitType.ASTRAL  # assumed astral per wiki
        self.action_priority = Priority.PRIORITY_6  # very low priority
        self.remembered = False

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not game or not target:
            return CANCEL_ACTION_MESSAGE

        if target.is_alive:
            return get_role_message(self.name, "remember_target_alive", Perspective.SELF)

        # Prevent duplicate remembers of unique roles
        if target.role.is_unique:
            # If any living player already has that role, fail.
            if any(p.is_alive and p.role.name == target.role.name for p in game.players):
                return get_role_message(self.name, "remember_unique_role_taken", Perspective.SELF, role_name=target.role.name.value)

        # Edge-cases: Godfather / Coven Leader / Pestilence special downgrades
        remembered_role_name = target.role.name
        if remembered_role_name == RoleName.GODFATHER:
            mafia_alive = any(p.is_alive and p.role.faction == Faction.MAFIA and p.role.name != RoleName.GODFATHER for p in game.players)
            if not mafia_alive:
                remembered_role_name = RoleName.MAFIOSO
        elif remembered_role_name == RoleName.COVEN_LEADER:
            coven_alive = any(p.is_alive and p.role.faction == Faction.COVEN for p in game.players)
            if coven_alive:
                # Guaranteed Necronomicon later – handled by existing nighttime allocation
                pass
        elif remembered_role_name == RoleName.PESTILENCE:
            remembered_role_name = RoleName.PLAGUEBEARER

        # Mark the intention – actual switch happens at dawn so defense doesn't apply tonight
        player.remember_role_name = remembered_role_name
        game.chat.add_player_notification(player, get_role_message(self.name, "remember_success_notification", Perspective.SELF, role_name=remembered_role_name.value))
        return get_role_message(self.name, "remember_success_action", Perspective.SELF, role_name=remembered_role_name.value)

class Ambusher(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.AMBUSHER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.PRIORITY_1  #pick ambush location
        self.attack = Attack.BASIC
        self.visit_type = VisitType.HARMFUL
        self.ambush_location: 'Player' | None = None

    def perform_night_action(self, player: 'Player', target: 'Player', game: 'Game' = None):
        if not target or not game:
            return CANCEL_ACTION_MESSAGE
        self.ambush_location = target
        return get_role_message(self.name, "ambush_target", Perspective.SELF, target_name=target.name)

class Blackmailer(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.BLACKMAILER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.PRIORITY_3

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not game or not target:
            return CANCEL_ACTION_MESSAGE

        target.is_blackmailed = True
        game.chat.add_player_notification(target, get_role_message(self.name, "blackmailed_notification", Perspective.OTHER))
        return get_role_message(self.name, "blackmail_success", Perspective.SELF, target_name=target.name)

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
        self.action_priority = Priority.PRIORITY_2
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
            return get_role_message(self.name, "no_assigned_target", Perspective.SELF)
        self.protect_target.protected_by.append(player)
        return get_role_message(self.name, "shield_target", Perspective.SELF, target_name=self.protect_target.name)

class Survivor(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.SURVIVOR
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.vests = 4
        self.visit_type = VisitType.ASTRAL
        self.action_priority = Priority.PRIORITY_3

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if self.vests > 0:
            self.vests -= 1
            player.defense = Defense.BASIC
            return get_role_message(self.name, "use_vest_success", Perspective.SELF, vests_left=self.vests)
        return get_role_message(self.name, "use_vest_no_charges", Perspective.SELF)

class Executioner(Role):
    """Neutral Evil role whose sole goal is to get their randomly-assigned Town target lynched."""

    def __init__(self):
        super().__init__()
        self.name = RoleName.EXECUTIONER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.PRIORITY_6  # very low – no real night action
        self.defense = Defense.BASIC
        self.detection_immune = True
        self.visit_type = VisitType.NON_HARMFUL

        # The Town player the Executioner wants lynched. Chosen at game start.
        self.target: 'Player' | None = None

    def assign_target(self, game: 'Game'):
        """Pick a random Town player other than self. Called by Game at init."""
        town_candidates = [p for p in game.players
                           if p.role.faction == Faction.TOWN
                           and p.name != self.name
                           and p.role.name not in [RoleName.MAYOR, RoleName.JAILOR]]
        if town_candidates:
            self.target = random.choice(town_candidates)

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        """Executioner has no night power; return a flavour string for logs."""
        return get_role_message(self.name, "no_night_action", Perspective.SELF)

class Jester(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.JESTER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_roleblock_immune = False
        self.detection_immune = True
        self.action_priority = Priority.DAY_ACTION #Haunting is a result of a day lynch
        self.astral_immune = True  # simple flag for future use

    def perform_day_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        #This is for haunting. It should only be called if the jester is lynched.
        if not target:
            return CANCEL_ACTION_MESSAGE
        if not game:
            return CANCEL_ACTION_MESSAGE.get_message(self.name, Perspective.SELF, "game_state_unavailable")
        
        #Jester's haunt is an unstoppable attack
        game.register_attack(player, target, Attack.UNSTOPPABLE, is_haunt=True)
        return get_role_message(self.name, "jester_haunt_success", Perspective.SELF, target_name=target.name)

    def on_lynch(self, game: 'Game'):
        #In a real game, this would trigger the haunting ability.
        #The player would then choose a target from the players who voted guilty.
        #Here we'll simply log the event.
        print(f"JESTER WIN: {self.name} has been lynched and will haunt someone!")

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        pass

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
        self.control_immune = True
        self.action_priority = Priority.PRIORITY_2
        self.has_been_attacked = False

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        if not game:
            return CANCEL_ACTION_MESSAGE.get_message(self.name, Perspective.SELF, "game_state_unavailable")

        # Expect the player to have submitted a tuple (control_target, force_target)
        # through the InteractionHandler stored in game.night_actions.
        try:
            control_target, force_target = game.night_actions[player]
        except (KeyError, TypeError, ValueError):
            return CANCEL_ACTION_MESSAGE.get_message(self.name, Perspective.SELF, "invalid_target_format")

        if not control_target or not force_target:
            return CANCEL_ACTION_MESSAGE

        if control_target.role.control_immune:
            game.chat.add_player_notification(control_target, get_role_message(self.name, "control_immune_notification", Perspective.OTHER))
            return get_role_message(self.name, "control_immune_target", Perspective.SELF)

        # Apply control – replace their submitted action with the forced one
        game.night_actions[control_target] = force_target
        control_target.is_controlled = True
        game.chat.add_player_notification(control_target, get_role_message(self.name, "controlled_notification", Perspective.OTHER, target_name=force_target.name))

        msg = get_role_message(self.name, "control_success", Perspective.SELF, control_target_name=control_target.name, force_target_name=force_target.name)

        # Necronomicon bonus attack (Powerful) on the victim
        if self.has_necronomicon:
            game.register_attack(player, victim, Attack.POWERFUL)
            msg += get_role_message(self.name, "control_necronomicon_attack", Perspective.SELF, victim_name=victim.name)

        return msg

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
        self.action_priority = Priority.PRIORITY_5
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
        if not game:
            return CANCEL_ACTION_MESSAGE.get_message(self.name, Perspective.SELF, "game_state_unavailable")

        # Before first kill: only act on full-moon (even) nights
        if self.kills == 0 and game.day % 2 != 0:
            return get_role_message(self.name, "juggernaut_no_full_moon", Perspective.SELF)

        # If no target chosen, stay home (no effect)
        if not target:
            return CANCEL_ACTION_MESSAGE

        self._update_power()

        if self.kills >= 2:
            # Rampage at location – hit target and visitors
            game.register_attack(player, target, self.attack)
            for v in target.targeted_by:
                if v.is_alive:
                    game.register_attack(player, v, self.attack, is_primary=False)
            return get_role_message(self.name, "juggernaut_rampage", Perspective.SELF, target_name=target.name, attack_strength=self.attack.name.lower())
        else:
            game.register_attack(player, target, self.attack)
            return get_role_message(self.name, "juggernaut_strike", Perspective.SELF, target_name=target.name, attack_strength=self.attack.name.lower())

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
        self.action_priority = Priority.PRIORITY_5
        self.attack = Attack.POWERFUL
        self.visit_type = VisitType.HARMFUL  #Rampage at location
        self.full_moon_cycle = 2  #every 2nd night starting N2
        self.is_roleblock_immune = False  # Werewolf is not roleblock immune

    def _is_full_moon(self, game: 'Game'):
        #Night number is game.day; full moon nights are even-numbered nights (N2,4…)
        return game.day % self.full_moon_cycle == 0

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        """Implements full-moon rampage logic.

        • Full-moon check – only acts on even night numbers (N2, N4 …).
        • If jailed (but not executed) attack Jailor inside cell.
        • If no target or target is self ⇒ rampage at home.
        • Otherwise rampage at target's house, also hitting their visitors.
        • If the chosen target is jailed: rampage hits *visitors* to the jailed player instead.
        • If trying to skip on full moon, forced to rampage at home.
        """
        if not game:
            return CANCEL_ACTION_MESSAGE.get_message(self.name, Perspective.SELF, "game_state_unavailable")

        # Full moon gating
        if not self._is_full_moon(game):
            return get_role_message(self.name, "werewolf_no_full_moon", Perspective.SELF)

        # In jail – Jailor did not execute ⇒ strike Jailor
        if player.is_jailed:
            jailor = game.find_player_by_role(RoleName.JAILOR)
            if jailor and jailor.is_alive and not getattr(jailor.role, 'executing', False):
                game.register_attack(player, jailor, self.attack)
                return get_role_message(self.name, "werewolf_maul_jailor", Perspective.SELF)
            return get_role_message(self.name, "werewolf_executed_in_jail", Perspective.SELF)  # handled elsewhere

        # Check if trying to skip - on full moon, forced to rampage at home
        if target is None:
            # Stay home – attack all visitors (but not self)
            victims = [v for v in player.targeted_by if v.is_alive and v != player]
            for v in victims:
                game.register_attack(player, v, self.attack, is_primary=False)
            if victims:
                return get_role_message(self.name, "werewolf_rampage_home_visitors", Perspective.SELF, visitor_list=', '.join(v.name for v in victims))
            return get_role_message(self.name, "werewolf_rampage_home_no_visitors", Perspective.SELF)

        # Determine rampage location
        if target == player:
            # Stay home – attack all visitors (but not self)
            victims = [v for v in player.targeted_by if v.is_alive and v != player]
            for v in victims:
                game.register_attack(player, v, self.attack, is_primary=False)
            if victims:
                return get_role_message(self.name, "werewolf_rampage_home_visitors", Perspective.SELF, visitor_list=', '.join(v.name for v in victims))
            return get_role_message(self.name, "werewolf_rampage_home_no_visitors", Perspective.SELF)

        # If target jailed, hit visitors only (not Jailor or prisoner)
        if target.is_jailed:
            visitors = [v for v in target.targeted_by if v.is_alive and v != player]
            for v in visitors:
                game.register_attack(player, v, self.attack, is_primary=False)
            return get_role_message(self.name, "werewolf_rampage_jailed_target", Perspective.SELF, target_name=target.name)

        # Normal rampage at target's house - attack target and ALL visitors
        game.register_attack(player, target, self.attack)
        for v in target.targeted_by:
            if v.is_alive and v != player:  # Don't attack self
                game.register_attack(player, v, self.attack, is_primary=False)
        return get_role_message(self.name, "werewolf_rampage_target", Perspective.SELF, target_name=target.name)

#Neutral Chaos – Plaguebearer / Pestilence
class Plaguebearer(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.PLAGUEBEARER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.infected = set()
        self.action_priority = Priority.PRIORITY_3

    def perform_night_action(self, player:'Player', target:'Player'=None, game:'Game'=None):
        if not game:
            return CANCEL_ACTION_MESSAGE.get_message(self.name, Perspective.SELF, "game_state_unavailable")

        # If targeting self or no target, stay home and infect visitors
        if not target or target == player:
            for visitor in player.targeted_by:
                if visitor.is_alive and not visitor.is_infected:
                    visitor.is_infected = True
                    game.chat.add_player_notification(visitor, get_role_message(self.name, "infected_notification", Perspective.OTHER))
                    print(f"[Plaguebearer] {visitor.name} was infected by visiting {player.name}.")
            result = get_role_message(self.name, "plaguebearer_infect_visitors", Perspective.SELF)
        else:
            # Visit target and infect them + their visitors
            if not target.is_infected:
                target.is_infected = True
                game.chat.add_player_notification(target, get_role_message(self.name, "infected_notification", Perspective.OTHER))
                print(f"[Plaguebearer] {player.name} infected {target.name}.")
            
            # Also infect anyone visiting the target
            for visitor in target.targeted_by:
                if visitor.is_alive and not visitor.is_infected:
                    visitor.is_infected = True
                    game.chat.add_player_notification(visitor, get_role_message(self.name, "infected_notification", Perspective.OTHER))
                    print(f"[Plaguebearer] {visitor.name} was infected by visiting {target.name}.")
            
            result = get_role_message(self.name, "plaguebearer_infect_target_and_visitors", Perspective.SELF, target_name=target.name)

        # Check if all others are infected (transformation condition)
        others = [p for p in game.players if p != player and p.is_alive]
        if others and all(p.is_infected for p in others):
            player.assign_role(Pestilence())
            game.chat.add_player_notification(player, get_role_message(self.name, "plaguebearer_transform_pestilence", Perspective.SELF))
            return get_role_message(self.name, "plaguebearer_transform_pestilence_action", Perspective.SELF)

        # Message feedback
        if target and target != player:
            return get_role_message(self.name, "plaguebearer_infect_target_and_visitors", Perspective.SELF, target_name=target.name)
        else:
            return get_role_message(self.name, "plaguebearer_infect_visitors", Perspective.SELF)

class Pestilence(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.PESTILENCE
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.attack = Attack.UNSTOPPABLE
        self.defense = Defense.INVINCIBLE
        self.action_priority = Priority.PRIORITY_5
        self.visit_type = VisitType.HARMFUL

    def perform_night_action(self, player:'Player', target:'Player'=None, game:'Game'=None):
        if not game or not target:
            return CANCEL_ACTION_MESSAGE
        #Rampage like Werewolf every night – attack target and everyone who visits them.
        #Pestilence should NEVER damage itself, even if it chooses to remain at home.
        
        # If the selected target is the Pestilence itself (stay-home rampage), only strike visitors.
        if target == player:
            for v in player.targeted_by:
                if v.is_alive and v != player:
                    game.register_attack(player, v, self.attack, is_primary=False)
            return get_role_message(self.name, "pestilence_rampage_home", Perspective.SELF)

        # Normal rampage on another player
        game.register_attack(player, target, self.attack)
        for v in target.targeted_by:
            if v.is_alive and v != player:  # exclude self to avoid suicide
                game.register_attack(player, v, self.attack, is_primary=False)
        return get_role_message(self.name, "pestilence_rampage_target", Perspective.SELF, target_name=target.name)

#Neutral Chaos – Vampire
class Vampire(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.VAMPIRE
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.action_priority = Priority.PRIORITY_5
        self.visit_type = VisitType.NON_HARMFUL
        self.detection_immune = True
        self.attack = Attack.BASIC  # Attack strength when biting non-convertible target or >4 vamps
        self.bite_cooldown = 0  # Must wait a night after successful conversion
        self.youngest_vampire = False  # Track if this is the youngest vampire

    def perform_night_action(self, player:'Player', target:'Player'=None, game:'Game'=None):
        if not game:
            return CANCEL_ACTION_MESSAGE.get_message(self.name, Perspective.SELF, "game_state_unavailable")
        
        # Check if on cooldown
        if self.bite_cooldown > 0:
            self.bite_cooldown -= 1
            return get_role_message(self.name, "bite_cooldown", Perspective.SELF)
        
        # Vampire voting system - collect all vampire votes
        vampires = [p for p in game.players if p.is_alive and p.role.name == RoleName.VAMPIRE]
        vampire_votes = {}
        
        # Collect votes from all vampires
        for vamp in vampires:
            vamp_target = game.night_actions.get(vamp)
            if vamp_target and vamp_target != player:  # Don't count self-votes
                if vamp_target not in vampire_votes:
                    vampire_votes[vamp_target] = 0
                vampire_votes[vamp_target] += 1
        
        if not vampire_votes:
            return get_role_message(self.name, "no_vampire_votes", Perspective.SELF)
        
        # Determine target based on weighted random selection
        import random
        total_votes = sum(vampire_votes.values())
        targets_and_weights = [(target, votes/total_votes) for target, votes in vampire_votes.items()]
        
        # Weighted random selection
        rand = random.random()
        cumulative = 0
        selected_target = None
        for target, weight in targets_and_weights:
            cumulative += weight
            if rand <= cumulative:
                selected_target = target
                break
        
        if not selected_target:
            selected_target = random.choice(list(vampire_votes.keys()))
        
        # Find youngest vampire to perform the bite
        youngest = min(vampires, key=lambda v: getattr(v, 'vampire_age', 0))
        
        # Only the youngest vampire actually performs the visit
        if player != youngest:
            return get_role_message(self.name, "youngest_vampire_bites", Perspective.SELF, youngest_name=youngest.name, target_name=selected_target.name)
        
        # Youngest vampire performs the bite
        return self._perform_bite(player, selected_target, game, vampires)
    
    def _perform_bite(self, player, target, game, vampires):
        """Perform the actual bite attempt."""
        vampire_count = len(vampires)
        
        # Check target's role and defense
        if target.role.name == RoleName.VAMPIRE_HUNTER:
            # Vampire Hunter kills the biting vampire
            game.register_attack(target, player, Attack.BASIC)
            return get_role_message(self.name, "bite_vampire_hunter_killed", Perspective.SELF, target_name=target.name)
        
        if target.role.name == RoleName.VAMPIRE:
            return get_role_message(self.name, "bite_target_already_vampire", Perspective.SELF, target_name=target.name)
        
        # Check if target has defense
        if target.defense != Defense.NONE:
            return get_role_message(self.name, "bite_target_defense_too_strong", Perspective.SELF, target_name=target.name)
        
        # Check vampire count - if 4+ vampires, kill instead of convert
        if vampire_count >= 4:
            # Kill non-convertible targets or when at max capacity
            convertible_roles = {
                RoleName.INVESTIGATOR, RoleName.LOOKOUT, RoleName.SHERIFF, RoleName.SPY, RoleName.PSYCHIC, RoleName.TRACKER,
                RoleName.BODYGUARD, RoleName.CRUSADER, RoleName.DOCTOR, RoleName.TRAPPER,
                RoleName.VIGILANTE, RoleName.VETERAN, RoleName.VAMPIRE_HUNTER,
                RoleName.MAYOR, RoleName.MEDIUM, RoleName.RETRIBUTIONIST, RoleName.TAVERN_KEEPER, RoleName.TRANSPORTER,
                RoleName.SURVIVOR, RoleName.AMNESIAC, RoleName.JESTER
            }
            
            if target.role.name not in convertible_roles or target.role.faction in [Faction.MAFIA, Faction.COVEN]:
                game.register_attack(player, target, self.attack)
                # Reset cooldown for all vampires since they killed
                for vamp in vampires:
                    vamp.role.bite_cooldown = 0
                return get_role_message(self.name, "bite_killed_non_convertible_or_max_vamps", Perspective.SELF, target_name=target.name)
            else:
                game.register_attack(player, target, self.attack)
                for vamp in vampires:
                    vamp.role.bite_cooldown = 0
                return get_role_message(self.name, "bite_killed_vampire_capacity_reached", Perspective.SELF, target_name=target.name)
        
        # Check if target is convertible
        convertible_roles = {
            RoleName.INVESTIGATOR, RoleName.LOOKOUT, RoleName.SHERIFF, RoleName.SPY, RoleName.PSYCHIC, RoleName.TRACKER,
            RoleName.BODYGUARD, RoleName.CRUSADER, RoleName.DOCTOR, RoleName.TRAPPER,
            RoleName.VIGILANTE, RoleName.VETERAN, RoleName.VAMPIRE_HUNTER,
            RoleName.MAYOR, RoleName.MEDIUM, RoleName.RETRIBUTIONIST, RoleName.TAVERN_KEEPER, RoleName.TRANSPORTER,
            RoleName.SURVIVOR, RoleName.AMNESIAC, RoleName.JESTER
        }
        
        # Non-convertible roles get killed
        if target.role.name not in convertible_roles or target.role.faction in [Faction.MAFIA, Faction.COVEN]:
            game.register_attack(player, target, self.attack)
            # Reset cooldown for all vampires since they killed
            for vamp in vampires:
                vamp.role.bite_cooldown = 0
            return get_role_message(self.name, "bite_killed_non_convertible_role", Perspective.SELF, target_name=target.name)
        
        # Success - convert to vampire
        target.assign_role(Vampire())
        target.role.vampire_age = max(getattr(v.role, 'vampire_age', 0) for v in vampires) + 1
        game.chat.add_player_notification(target, get_role_message(self.name, "converted_to_vampire_notification", Perspective.OTHER))
        
        # Set cooldown for all vampires
        for vamp in vampires + [target]:
            vamp.role.bite_cooldown = 1
        
        # Notify all vampires of the bite
        bite_msg = get_role_message(self.name, "vampire_visited_target", Perspective.OTHER, target_name=target.name)
        for vamp in vampires + [target]:
            game.chat.add_player_notification(vamp, bite_msg)
        
        print(f"[Vampire] {target.name} has been converted to a Vampire!")
        return get_role_message(self.name, "bite_success", Perspective.SELF, target_name=target.name)

class CovenLeader(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.COVEN_LEADER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_coven = True
        self.action_priority = Priority.PRIORITY_2
        self.is_roleblock_immune = True
        self.control_immune = True
        self.visit_type = VisitType.HARMFUL

    def perform_night_action(self, player:'Player', target:'Player'=None, game:'Game'=None):
        if not game:
            return None

        # InteractionHandler always stores tuple (puppet, victim)
        try:
            puppet, victim = game.night_actions[player]
        except (KeyError, TypeError, ValueError):
            return "You must select two targets to direct your magic."

        if not puppet or not victim:
            return "You must select two targets."

        # Handle control immunity
        if puppet.role.control_immune:
            puppet.notifications.append("Someone tried to control you but you were immune!")
            return "Your puppet resisted your control!"

        # Apply control – overwrite puppet's submitted action (or add one) with forced victim
        game.night_actions[puppet] = victim
        puppet.is_controlled = True
        puppet.notifications.append(f"You were controlled by dark magic! You were forced to target {victim.name}.")

        msg = f"You controlled {puppet.name} to target {victim.name}."

        # Necronomicon bonus attack (Powerful) on the victim
        if self.has_necronomicon:
            game.register_attack(player, victim, Attack.POWERFUL)
            msg += f" Using the Necronomicon, you also attack {victim.name}."

        return msg

class Medusa(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.MEDUSA
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_coven = True
        self.visit_type = VisitType.ASTRAL
        self.action_priority = Priority.PRIORITY_3  #stone gaze before investigator
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
        self.action_priority = Priority.PRIORITY_3
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
        self.action_priority = Priority.PRIORITY_3
        # Track individual potion cooldowns (heal/reveal/attack)
        self._cooldowns = {"heal": 0, "reveal": 0, "attack": 0}
        self._last_used = None  # last potion used

    def perform_night_action(self, player: 'Player', target: 'Player' = None, game: 'Game' = None):
        """Rotate through Heal → Reveal → Attack, each on a 3-night cooldown.

        With the Necronomicon, cooldowns are ignored.
        """
        if not game or not target:
            return "You chose not to use a potion."

        # Select next potion type respecting cooldowns
        order = ["heal", "reveal", "attack"]
        choice = None

        # If Necronomicon held, never any cooldown – just cycle
        if self.has_necronomicon:
            idx = 0 if self._last_used is None else (order.index(self._last_used) + 1) % 3
            choice = order[idx]
        else:
            # Find first potion whose cooldown is 0, in order heal->reveal->attack
            for pot in order:
                if self._cooldowns[pot] == 0:
                    choice = pot
                    break
            if choice is None:
                return "All of your potions are re-charging."  # edge case

        # Apply potion effect
        if choice == "attack":
            atk = Attack.BASIC if not self.has_necronomicon else Attack.POWERFUL
            game.register_attack(player, target, atk)
            result = f"You hurled a harmful potion at {target.name}."
        elif choice == "heal":
            target.defense = Defense.POWERFUL if self.has_necronomicon else Defense.BASIC
            target.notifications.append("You feel rejuvenated by powerful alchemical energies!")
            result = f"You healed {target.name}."
        else:  # reveal
            role_name = target.role.name.value
            for coven_member in [p for p in game.players if p.role.is_coven]:
                coven_member.notifications.append(f"Potion vision: {target.name} is a {role_name}.")
            result = f"You revealed {target.name}'s role to the Coven."

        # Update cooldowns (skip if Necronomicon)
        if not self.has_necronomicon:
            # Set chosen potion to cooldown 3, decrement others if >0
            for pot in order:
                if self._cooldowns[pot] > 0:
                    self._cooldowns[pot] -= 1
            self._cooldowns[choice] = 3

        self._last_used = choice
        return result

class Poisoner(Role):
    def __init__(self):
        super().__init__()
        self.name = RoleName.POISONER
        self.alignment = get_role_alignment(self.name)
        self.faction = get_role_faction(self.name)
        self.is_coven = True
        self.action_priority = Priority.PRIORITY_5

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
        self.is_unique = True
        self.is_roleblock_immune = True
        self.control_immune = True
        self.detection_immune = False
        self.action_priority = Priority.PRIORITY_1
        # Track used corpses so they "rot" after one night
        self._used_corpses: set['Player'] = set()
        from .enums import VisitType
        self.visit_type = VisitType.ASTRAL

    def perform_night_action(self, player: 'Player', target: 'tuple[Player, Player]' = None, game: 'Game' = None):
        """Expect *target* to be a tuple (corpse, victim).

        The selected corpse is temporarily re-animated to perform its own ability
        on the second target. Each corpse can only be used once.
        """
        if not game or not isinstance(target, tuple) or len(target) != 2:
            return "You must choose a corpse and a target (corpse,target)."

        corpse, victim = target

        if corpse.is_alive:
            return f"{corpse.name} is still alive – you can only reanimate the dead."
        if corpse in self._used_corpses:
            return f"{corpse.name}'s corpse has already rotted and cannot be used again."
        if corpse.was_cleaned:
            return f"{corpse.name}'s identity was hidden. You cannot reanimate this corpse."

        # Temporarily bring corpse to life to run its ability
        original_state = corpse.is_alive
        corpse.is_alive = True
        try:
            result = corpse.role.perform_night_action(corpse, victim, game)
        finally:
            corpse.is_alive = original_state

        self._used_corpses.add(corpse)

        # Register visit for lookout / tracker purposes
        corpse.visit(victim)

        return f"You reanimated {corpse.name}'s corpse – {result}"

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
    RoleName.AMNESIAC: Amnesiac,
    RoleName.AMBUSHER: Ambusher,
    RoleName.BLACKMAILER: Blackmailer,
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
    RoleName.POTION_MASTER: PotionMaster,
    RoleName.POISONER: Poisoner,
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

def create_role_from_name(role_name: RoleName) -> 'Role':
    """Factory function to create a role instance from a RoleName enum."""
    role_class_map = {
        RoleName.SHERIFF: Sheriff,
        RoleName.INVESTIGATOR: Investigator,
        RoleName.LOOKOUT: Lookout,
        RoleName.DOCTOR: Doctor,
        RoleName.BODYGUARD: Bodyguard,
        RoleName.CRUSADER: Crusader,
        RoleName.VIGILANTE: Vigilante,
        RoleName.MAYOR: Mayor,
        RoleName.MEDIUM: Medium,
        RoleName.TAVERN_KEEPER: TavernKeeper,
        RoleName.GODFATHER: Godfather,
        RoleName.MAFIOSO: Mafioso,
        RoleName.SERIAL_KILLER: SerialKiller,
        RoleName.JAILOR: Jailor,
        RoleName.CONSIGLIERE: Consigliere,
        RoleName.BOOTLEGGER: Bootlegger,
        RoleName.ARSONIST: Arsonist,
        RoleName.PIRATE: Pirate,
        RoleName.VETERAN: Veteran,
        RoleName.PSYCHIC: Psychic,
        RoleName.SPY: Spy,
        RoleName.TRACKER: Tracker,
        RoleName.TRAPPER: Trapper,
        RoleName.VAMPIRE_HUNTER: VampireHunter,
        RoleName.RETRIBUTIONIST: Retributionist,
        RoleName.TRANSPORTER: Transporter,
        RoleName.DISGUISER: Disguiser,
        RoleName.FORGER: Forger,
        RoleName.FRAMER: Framer,
        RoleName.HYPNOTIST: Hypnotist,
        RoleName.JANITOR: Janitor,
        RoleName.AMNESIAC: Amnesiac,
        RoleName.AMBUSHER: Ambusher,
        RoleName.BLACKMAILER: Blackmailer,
        RoleName.GUARDIAN_ANGEL: GuardianAngel,
        RoleName.SURVIVOR: Survivor,
        RoleName.EXECUTIONER: Executioner,
        RoleName.JESTER: Jester,
        RoleName.WITCH: Witch,
        RoleName.JUGGERNAUT: Juggernaut,
        RoleName.WEREWOLF: Werewolf,
        RoleName.PLAGUEBEARER: Plaguebearer,
        RoleName.PESTILENCE: Pestilence,
        RoleName.VAMPIRE: Vampire,
        RoleName.COVEN_LEADER: CovenLeader,
        RoleName.MEDUSA: Medusa,
        RoleName.POTION_MASTER: PotionMaster,
        RoleName.POISONER: Poisoner,
        RoleName.NECROMANCER: Necromancer,
        RoleName.HEX_MASTER: HexMaster,
    }
    # Handle aliases
    if role_name == RoleName.ESCORT:
        role_name = RoleName.TAVERN_KEEPER
    if role_name == RoleName.CONSORT:
        role_name = RoleName.BOOTLEGGER
        
    role_class = role_class_map.get(role_name)
    if role_class:
        return role_class()
    
    # Fallback for roles not fully implemented yet
    return PlaceholderRole(role_name) 
