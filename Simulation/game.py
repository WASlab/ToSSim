from .config import GameConfiguration
from .player import Player
from .enums import Faction, RoleName, Attack, Defense, Time, Phase, VisitType, Priority, ImmunityType
from .roles import Role, Arsonist, SerialKiller, has_immunity
import random
from collections import defaultdict

class Game:
    def __init__(self, config: GameConfiguration, players: list[Player]):
        self.config = config
        self.players = players
        self.day = 0
        self.time = Time.DAY
        self.phase = Phase.DAY
        self.graveyard = []
        self.day_actions = {}
        self.night_actions = {}
        self.night_attacks = []
        self.winners = []
        self.traps = []  #list of traps
        #assign targets to executioners
        for pl in self.players:
            if pl.role.name == RoleName.EXECUTIONER and pl.role.target is None:
                pl.role.assign_target(self)
            if pl.role.name == RoleName.GUARDIAN_ANGEL and getattr(pl.role, 'protect_target', None) is None:
                pl.role.assign_protect_target(self)

    def get_player_by_name(self, name: str) -> Player | None:
        for player in self.players:
            if player.name == name:
                return player
        return None

    def find_player_by_role(self, role_name: RoleName) -> Player | None:
        """Finds the first living player with a given role."""
        for player in self.players:
            if player.is_alive and player.role.name == role_name:
                return player
        return None

    def run_game(self):
        while self.game_is_over() is None:
            self.day += 1
            print(f"\n----- Day {self.day} -----")
            self.run_day()
            
            if self.game_is_over() is not None:
                break
                
            print(f"\n----- Night {self.day} -----")
            self.run_night()

        winner = self.game_is_over()
        print(f"\n--- Game Over ---")
        print(f"The {winner.name} faction has won!")

    def run_day(self):
        print("All players wake up.")
        for player in self.players:
            player.reset_night_states()

        #Day action submission phase
        self._simulate_day_actions()
        self._process_day_actions()

        #Discussion and Voting Phase (Simplified)
        print("Begin discussion and voting.")

    def run_night(self):
        #Night action submission phase
        self._assign_necronomicon()
        self._simulate_night_actions()

        #Establish visits
        self._process_visits()

        #Resolve traps before other abilities
        self._process_traps()

        #Transporters act first and can change the outcome of all other actions
        self._process_transports()

        #Process abilities that trigger on visit (passives)
        self._process_passive_abilities()

        #Process lingering poison deaths
        self._process_poison()

        #Process all submitted night actions by priority
        self._process_night_actions()

        #Resolve all registered attacks
        self._process_attacks()
        
        #Hex Master final hex wipe
        self._check_final_hex()

        #Announce deaths and notifications
        self._announce_deaths()
        self._send_notifications()

    def submit_day_action(self, player: Player, target: Player = None):
        self.day_actions[player] = target

    def submit_night_action(self, player: Player, target: Player = None):
        self.night_actions[player] = target

    def register_attack(
        self, attacker: Player, target: Player, attack_type: Attack, 
        is_primary: bool = True, is_duel_win: bool = False
    ):
        self.night_attacks.append({
            "attacker": attacker,
            "target": target,
            "type": attack_type,
            "is_primary": is_primary,
            "is_duel_win": is_duel_win
        })

    def ignite_doused_players(self, igniter: Player):
        for player in self.players:
            if player.is_doused:
                self.register_attack(igniter, player, Attack.UNSTOPPABLE)

    def _process_day_actions(self):
        for player, target in self.day_actions.items():
            if player.is_alive:
                result = player.role.perform_day_action(player, target, self)
                if result:
                    print(f"[Day Action] {player.name}: {result}")
        self.day_actions.clear()

    def _process_visits(self):
        #Clear old visits first
        for player in self.players:
            player.targeted_by = []
        
        for player, target in self.night_actions.items():
            if player.is_alive and target:
                if player.role.visit_type != VisitType.ASTRAL:
                    player.visit(target)

    def _process_traps(self):
        """Handle Trapper traps set this night."""
        if not self.traps:
            return
        from .enums import VisitType
        for trap in list(self.traps):
            if not trap["active"]:
                continue
            location = trap["location"]
            harmful_visitors = [v for v in location.targeted_by if v.role.visit_type == VisitType.HARMFUL and v.is_alive]
            if harmful_visitors:
                victim = harmful_visitors[0]
                owner = trap["owner"]
                self.register_attack(owner, victim, Attack.POWERFUL, is_primary=False)
                print(f"[Trap] {owner.name}'s trap triggered on {victim.name} while visiting {location.name}.")
                trap["active"] = False
        #Remove used traps
        self.traps = [t for t in self.traps if t["active"]]

    def _process_passive_abilities(self):
        for player in self.players:
            if not player.is_alive: continue

            if isinstance(player.role, Arsonist):
                for visitor in player.targeted_by:
                    if visitor.is_alive:
                        visitor.is_doused = True
                        print(f"[Passive] {visitor.name} was doused by visiting {player.name} (Arsonist).")

            if isinstance(player.role, SerialKiller):
                if player.is_jailed:
                    if not player.role.cautious:
                        jailor = self.find_player_by_role(RoleName.JAILOR)
                        if jailor and not player.is_being_executed:
                            self.register_attack(player, jailor, player.role.attack, is_primary=False)
                            print(f"[Passive] The jailed {player.name} (SK) attacks the Jailor.")
                    #Clear any other action the jailed SK might have had
                    if player in self.night_actions:
                         del self.night_actions[player]
                    continue

                if not player.role.cautious:
                    for visitor in player.targeted_by:
                        if visitor.is_alive and visitor.role.name in [RoleName.TAVERN_KEEPER, RoleName.BOOTLEGGER, RoleName.PIRATE]:
                            self.register_attack(player, visitor, player.role.attack, is_primary=False)
                            visitor.last_will_bloodied = True
                            print(f"[Passive] {player.name} (SK) also attacked {visitor.name} for visiting.")

            #Crusader protection – strike first visitor of different faction
            if player.role.__class__.__name__ == "Crusader":
                if player.visiting:  #crusader guards someone
                    guard_target = player.visiting
                    for visitor in guard_target.targeted_by:
                        if visitor == player:
                            continue
                        if visitor.role.faction != player.role.faction and visitor.is_alive:
                            self.register_attack(player, visitor, Attack.POWERFUL, is_primary=False)
                            print(f"[Crusader] {player.name} struck down {visitor.name} for visiting {guard_target.name}.")
                            break

            #Ambusher strike: after visits, attack first visitor to ambush location
            if player.role.__class__.__name__ == "Ambusher" and player.role.ambush_location:
                loc = player.role.ambush_location
                visitors = [v for v in loc.targeted_by if v != player and v.is_alive]
                if visitors:
                    victim = visitors[0]
                    self.register_attack(player, victim, Attack.BASIC, is_primary=False)
                    print(f"[Ambusher] {player.name} ambushed {victim.name} visiting {loc.name}!")
                    #Reveal name to all visitors
                    for v in visitors:
                        v.notifications.append(f"You saw {player.name} ambush someone at {loc.name}!")

    def _process_night_actions(self):
        sorted_actions = sorted(self.night_actions.items(), key=lambda item: item[0].role.action_priority.value)
        
        for player, target in sorted_actions:
            if not player.is_alive:
                continue

            #If the player is role-blocked but has immunity, the action still proceeds.
            if player.is_role_blocked and not has_immunity(player.role, ImmunityType.ROLE_BLOCK):
                continue

            kwargs = {}
            if isinstance(player.role, SerialKiller):
                kwargs['cautious'] = player.role.cautious
            result = player.role.perform_night_action(player, target, self, **kwargs)
            if result:
                print(f"[Night Action] {player.name}: {result}")

    def _process_attacks(self):
        #Handle Pirate vs SK duel priority
        #If a Pirate wins a duel against an SK, the SK's attack should be cancelled.
        pirate_duel_wins = [
            attack for attack in self.night_attacks 
            if attack["is_duel_win"] and isinstance(attack["target"].role, SerialKiller)
        ]

        #In this simplified model the Serial Killer should still attack the Pirate even if the Pirate wins.
        #Therefore we no longer cancel the SK→Pirate attack.

        #Process attacks
        for attack in sorted(self.night_attacks, key=lambda x: x['type'].value, reverse=True):
            attacker = attack["attacker"]
            target = attack["target"]

            #Re-check if attacker is alive, as they could have been killed by a higher-priority attack
            if not attacker.is_alive:
                continue
            
            #If target is already dead, skip
            if not target.is_alive:
                continue

            #Check for protection (Doctor, BG, etc.)
            if self._is_protected(target, attacker, attack["type"]):
                #Award plunder for successful duel even if saved
                if attack["is_duel_win"]:
                    if hasattr(attacker.role, 'plunders'):
                        attacker.role.plunders += 1
                        print(f"[Game Log] {attacker.name} won the duel against {target.name}, but they were protected. Plunder awarded.")
                continue

            if attack["type"].value > target.defense.value:
                target.is_alive = False
                print(f"[Game Log] {attacker.name} ({attacker.role.name.value}) killed {target.name} ({target.role.name.value})")
                #Vigilante guilt: mark if townie killed
                if attacker.role.name == RoleName.VIGILANTE and target.role.faction == Faction.TOWN:
                    attacker.role.has_killed_townie = True
                self.graveyard.append(target)

                #Handle Disguiser: if someone earlier set disguise_target to this now-dead body,
                #mark their apparent role as that of the corpse (simplified model)
                for p in self.players:
                    if hasattr(p, 'disguise_target') and p.disguise_target == target and p.is_alive:
                        #Save original role for future if needed, then switch
                        p.original_role_name = p.role.name
                        p.role.name = target.role.name
                        print(f"[Disguiser] {p.name} has disguised themselves as {target.name}!")

                #A won duel that results in a kill is a successful plunder.
                if attack["is_duel_win"]:
                    if hasattr(attacker.role, 'plunders'):
                        attacker.role.plunders += 1
                        print(f"[Game Log] {attacker.name} successfully plundered {target.name}.")

                #Disguiser logic – if any living player had set disguise_target to this victim, change their role label
                for pl in self.players:
                    if getattr(pl, 'disguise_target', None) == target and pl.is_alive:
                        pl.role.name = target.role.name
                        pl.disguise_success = True
                        print(f"[Disguiser] {pl.name} now appears as {target.role.name.value}.")

            else:
                 #Award plunder for successful duel even if defense was too high
                if attack["is_duel_win"]:
                    if hasattr(attacker.role, 'plunders'):
                        attacker.role.plunders += 1
                        print(f"[Game Log] {attacker.name} won the duel against {target.name}, but their defense was too strong. Plunder awarded.")


        #Clear attacks for the next night
        self.night_attacks = []

        #Check for individual win conditions
        for player in self.players:
            if player.is_alive and hasattr(player.role, 'plunders') and player.role.plunders >= 2:
                self.winners.append(player)
                print(f"[Game Log] Pirate {player.name} has won by getting 2 plunders!")
                return True

        #Check for factional win conditions
        alive_players = [p for p in self.players if p.is_alive]
        if not alive_players:
            return True

        #If any winners have been declared, the game is over
        if self.winners:
            return True

        return False

    def _announce_deaths(self):
        newly_dead = []
        for player in self.players:
            if not player.is_alive and player not in self.graveyard:
                newly_dead.append(player)
        
        if not newly_dead:
            print("No one died last night.")
        else:
            for dead_player in newly_dead:
                print(f"{dead_player.name} ({dead_player.role.name.value}) was found dead.")
                self.graveyard.append(dead_player)
        
        #Update Executioner win/transform after deaths recorded
        self._check_executioners(newly_dead)

    def _send_notifications(self):
        for player in self.players:
            if player.is_alive and player.notifications:
                for notif in player.notifications:
                    print(f"[Notification for {player.name}] {notif}")
        
        #Spy notifications
        spies = [p for p in self.players if p.is_alive and p.role.name == RoleName.SPY]
        if spies:
            mafia_visits = {p.name for p, t in self.night_actions.items() if p.role.faction == Faction.MAFIA and t is not None}
            coven_visits = {p.name for p, t in self.night_actions.items() if p.role.faction == Faction.COVEN and t is not None}

            if mafia_visits:
                for spy in spies:
                    spy.notifications.append(f"The Mafia visited: {', '.join(mafia_visits)}")
            if coven_visits:
                for spy in spies:
                    spy.notifications.append(f"The Coven visited: {', '.join(coven_visits)}")

    def game_is_over(self) -> Faction | None:
        living_players = [p for p in self.players if p.is_alive]
        factions = set(p.role.faction for p in living_players)
        if len(factions) == 1: return list(factions)[0]
        if Faction.MAFIA not in factions and Faction.NEUTRAL not in factions and Faction.COVEN not in factions:
            return Faction.TOWN
        return None

    def _simulate_day_actions(self):
        self.day_actions.clear()
        for player in self.players:
            if player.is_alive:
                if player.role.name == RoleName.JAILOR or player.role.name == RoleName.PIRATE:
                     targettable = [p for p in self.players if p.is_alive and p != player]
                     if targettable:
                         target = random.choice(targettable)
                         player.role.perform_day_action(player, target, self)

    def _simulate_night_actions(self):
        self.night_actions.clear()
        #Vigilante guilt-suicide: if a Vigilante killed a Townie, they shoot themselves tonight
        for v_player in self.players:
            if v_player.is_alive and v_player.role.name == RoleName.VIGILANTE and v_player.role.has_killed_townie and not v_player.role.put_gun_down:
                self.register_attack(v_player, v_player, Attack.UNSTOPPABLE)
                v_player.role.put_gun_down = True

        #Jailor's action is decided during the day
        jailed_player = None
        jailor = self.find_player_by_role(RoleName.JAILOR)
        if jailor and jailor.role.jailed_target:
            jailed_player = jailor.role.jailed_target

        for player in self.players:
            if player.is_alive and player != jailed_player:
                if player.role.name == RoleName.PIRATE: continue #Pirate action decided during day

                targettable = [p for p in self.players if p.is_alive]

                if player.role.name == RoleName.TRANSPORTER:
                    if len(targettable) >= 2:
                        target1, target2 = random.sample(targettable, 2)
                        self.submit_night_action(player, (target1, target2))
                    continue
                
                if isinstance(player.role, SerialKiller):
                    player.role.cautious = random.random() < 0.5

                if player.role.name == RoleName.ARSONIST:
                    if random.random() < 0.1: self.submit_night_action(player, player); continue
                    elif random.random() < 0.1: self.submit_night_action(player, None); continue
                
                if player.role.name in [RoleName.DOCTOR, RoleName.BODYGUARD]:
                     if random.random() < 0.1: self.submit_night_action(player, player); continue
                
                if player in targettable: targettable.remove(player)
                if jailed_player in targettable: targettable.remove(jailed_player)

                if targettable:
                    target = random.choice(targettable)
                    self.submit_night_action(player, target)

    def print_results(self):
        if self.winners:
            print("--- WINNERS ---")
            for winner in self.winners:
                print(f"{winner.name} ({winner.role.name.value})")
        else:
            #Determine winning faction if no individual winner
            winning_faction = self.game_is_over()
            if winning_faction:
                print(f"\nWinning Faction: {winning_faction.name}")
                for player in self.players:
                    if player.role.faction == winning_faction:
                        print(f"- {player.name} ({player.role.name.value})")
            else:
                print("\nDraw! No one wins.")

    def _is_protected(self, target: Player, attacker: Player, attack_type: Attack) -> bool:
        """Return True if the attack is prevented by a protector.

        Currently handles Bodyguard interception and simple defense buffs (already managed by defense values).
        """
        if not target.protected_by:
            return False

        #Bodyguards intercept first.
        #Guardian Angel shield – absolute save (beats unstoppable) without dying
        ga_protectors = [p for p in target.protected_by if p.is_alive and p.role.__class__.__name__ == "GuardianAngel"]
        if ga_protectors:
            print(f"[GuardianAngel] {ga_protectors[0].name} shielded {target.name} from harm!")
            return True

        interceptors = [p for p in target.protected_by if p.is_alive and p.role.__class__.__name__ == "Bodyguard" and attacker.role.visit_type == VisitType.HARMFUL]
        if interceptors:
            bg = interceptors[0]  #Only first BG intercepts in official rules

            #BG sacrifices themselves to protect.
            bg.is_alive = False
            self.graveyard.append(bg)
            print(f"[Protection] {bg.name} (Bodyguard) died protecting {target.name}!")

            #BG counterattacks the attacker unless the attack is Unstoppable.
            if attack_type != Attack.UNSTOPPABLE and attacker.is_alive:
                self.register_attack(bg, attacker, Attack.POWERFUL, is_primary=False)
                print(f"[Protection] {bg.name} struck back at {attacker.name}!")

            return True  #Target is saved

        #Future: Crusader, Guardian Angel, etc.
        return False

    def _process_transports(self):
        transporter_actions = {p: t for p, t in self.night_actions.items() if p.is_alive and p.role.name == RoleName.TRANSPORTER}
        
        if not transporter_actions:
            return

        #For simplicity, this implementation only handles one transporter.
        #Town of Salem games typically only allow one.
        if len(transporter_actions) > 1:
            print("[Warning] Multiple transporters are not yet supported. Only one will act.")

        #Get the two targets of the first transporter found
        #This assumes the transporter's target is a tuple of two players.
        #The simulation logic will need to be updated to support this.
        
        try:
            transporter, (target1, target2) = list(transporter_actions.items())[0]
        except (ValueError, TypeError):
             #This can happen if the target is not a tuple of two players.
             #We'll add a check for this in the simulation logic later.
            return

        if not all([target1, target2]):
            return

        print(f"[Transport] {transporter.name} is transporting {target1.name} and {target2.name}.")
        
        #Swap targets for all other actions
        for player, target in self.night_actions.items():
            if player == transporter: continue
            if target == target1:
                self.night_actions[player] = target2
                print(f"[Transport] {player.name}'s action was redirected to {target2.name}.")
            elif target == target2:
                self.night_actions[player] = target1
                print(f"[Transport] {player.name}'s action was redirected to {target1.name}.")
        
        #Also swap any attacks targeting them
        for attack in self.night_attacks:
            if attack["target"] == target1:
                attack["target"] = target2
            elif attack["target"] == target2:
                attack["target"] = target1
        
        #Notify the transported players
        target1.notifications.append("You were transported to another location.")
        target2.notifications.append("You were transported to another location.")

    def _check_executioners(self, dead_players):
        from .roles import Jester
        for p in self.players:
            if p.is_alive and p.role.name == RoleName.EXECUTIONER:
                ex_role = p.role
                if not ex_role.target:
                    continue
                if ex_role.target in dead_players:
                    if ex_role.target.was_lynched:
                        #Executioner wins immediately
                        self.winners.append(p)
                        print(f"[Executioner] {p.name} achieved their goal and wins individually!")
                    else:
                        #Becomes Jester
                        p.assign_role(Jester())
                        print(f"[Executioner] {p.name}'s target died unnaturally – they have become a Jester!")

    def _assign_necronomicon(self):
        """Give Necronomicon to coven each night if no one has it."""
        coven_alive = [p for p in self.players if p.is_alive and getattr(p.role,'is_coven',False)]
        if not coven_alive:
            return
        holder = next((p for p in coven_alive if p.role.has_necronomicon), None)
        if holder:
            return  #already held

        #inheritance order per wiki: Leader -> random coven (except Medusa) -> Medusa
        pick = None
        leader = next((p for p in coven_alive if p.role.name == RoleName.COVEN_LEADER), None)
        if leader:
            pick = leader
        else:
            non_medusa = [p for p in coven_alive if p.role.name != RoleName.MEDUSA]
            if non_medusa:
                import random as _r
                pick = _r.choice(non_medusa)
            else:
                #only Medusa remains
                medusa = next((p for p in coven_alive if p.role.name == RoleName.MEDUSA), None)
                if medusa:
                    pick = medusa
        if not pick:
            pick = coven_alive[0]

        #grant detection immunity while holding book
        pick.role.detection_immune = True
        pick.role.has_necronomicon = True
        pick.notifications.append("You have received the Necronomicon! Your powers are enhanced.")
        #Adjust action priority for certain roles that gain an attack with the Necronomicon
        if pick.role.name in [RoleName.MEDUSA, RoleName.COVEN_LEADER, RoleName.HEX_MASTER, RoleName.POTION_MASTER]:
            pick.role.action_priority = Priority.KILLING
        #boost attacks for simple roles
        print(f"[Necronomicon] {pick.name} now holds the Necronomicon.")

    def _process_poison(self):
        """Handle poison countdown and deaths."""
        for p in self.players:
            if not p.is_alive:
                continue
            if p.is_poisoned:
                if p.poison_timer == 0:
                    p.poison_timer = 1
                else:
                    #check for heal/purge
                    if p.poison_uncurable:
                        cured = False
                    else:
                        cured = any(pro.role.__class__.__name__ in ["Doctor","GuardianAngel","PotionMaster"] for pro in p.protected_by)
                    if cured:
                        p.is_poisoned = False
                        p.poison_timer = 0
                        p.notifications.append("You were cured of poison!")
                    else:
                        self.register_attack(p, p, Attack.BASIC)
                        p.is_poisoned = False
                        p.poison_timer = 0

    def _check_final_hex(self):
        hex_master_alive = [p for p in self.players if p.is_alive and p.role.name == RoleName.HEX_MASTER]
        if not hex_master_alive:
            return
        non_coven = [p for p in self.players if p.is_alive and not getattr(p.role,'is_coven',False)]
        if not non_coven:
            return
        all_hexed = all(p.is_hexed for p in non_coven)
        if all_hexed:
            for p in non_coven:
                self.register_attack(hex_master_alive[0], p, Attack.UNSTOPPABLE)
            print("[HexMaster] Final Hex unleashed! All non-Coven players perish.")