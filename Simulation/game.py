import random
from .player import Player
from .roles import (
    Role, Sheriff, Mafioso, SerialKiller, Investigator, Lookout, Spy, Psychic, Tracker,
    Godfather, Doctor, Bodyguard, Vigilante, Mayor, Medium, Escort
)
from .enums import RoleName, Faction, RoleAlignment, Attack, Defense
from .chat import ChatManager, ChatChannelType

class Game:
    def __init__(self, num_players: int, role_list: list[RoleName]):
        if num_players != len(role_list):
            raise ValueError("Number of players must match the number of roles in the role list.")
        
        self.players: list[Player] = [Player(f"Player {i}") for i in range(num_players)]
        self.roles: list[Role] = self._create_roles_from_list(role_list)
        self.day = 1
        self.is_day = True
        self.game_over = False
        self.winner: Faction | None = None
        self.graveyard = []
        self.chat_manager = ChatManager()

        self._assign_roles()
        self._setup_chat_channels()

    def _create_roles_from_list(self, role_list: list[RoleName]) -> list[Role]:
        # This mapping will need to be expanded as more roles are added.
        role_map = {
            RoleName.SHERIFF: Sheriff,
            RoleName.MAFIOSO: Mafioso,
            RoleName.SERIAL_KILLER: SerialKiller,
            RoleName.INVESTIGATOR: Investigator,
            RoleName.LOOKOUT: Lookout,
            RoleName.SPY: Spy,
            RoleName.PSYCHIC: Psychic,
            RoleName.TRACKER: Tracker,
            RoleName.GODFATHER: Godfather,
            RoleName.DOCTOR: Doctor,
            RoleName.BODYGUARD: Bodyguard,
            RoleName.VIGILANTE: Vigilante,
            RoleName.MAYOR: Mayor,
            RoleName.MEDIUM: Medium,
            RoleName.ESCORT: Escort,
        }
        return [role_map[role_name]() for role_name in role_list if role_name in role_map]

    def _assign_roles(self):
        random.shuffle(self.roles)
        for player, role in zip(self.players, self.roles):
            player.assign_role(role)

    def _setup_chat_channels(self):
        for player in self.players:
            self.chat_manager.add_player_to_channel(player, ChatChannelType.PUBLIC)
            if player.role.faction == Faction.MAFIA:
                self.chat_manager.add_player_to_channel(player, ChatChannelType.MAFIA)
        # Will add Coven, Vampire, etc. later

    def run_game(self):
        while not self.game_over:
            if self.is_day:
                self.run_day_phase()
            else:
                self.run_night_phase()
            self.check_win_conditions()
            self.day += 0.5 # A bit of a hack to increment day number after night
            self.is_day = not self.is_day

    def run_day_phase(self):
        print(f"--- Day {int(self.day)} ---")
        # Announce deaths from the night
        # For now, this is handled in the night phase printouts.

        # Check for dead Medium seance
        dead_mediums = [p for p in self.graveyard if p.role.name == RoleName.MEDIUM and p.role.seances > 0]
        for medium in dead_mediums:
            living_players = [p for p in self.players if p.is_alive]
            if living_players:
                target = random.choice(living_players)
                print(f"A message from beyond the grave for {target.name}: {medium.role.seance(target)}")

        print("Town is discussing...")
        self.run_chat_phase(ChatChannelType.PUBLIC)
        
        self.run_day_abilities()
        self.run_voting_phase()

        for player in self.players:
            player.clear_day_states()

    def run_day_abilities(self):
        print("--- Day Abilities ---")
        # Placeholder for Mayor reveal logic
        for player in self.players:
            if player.is_alive and player.role.name == RoleName.MAYOR and not player.role.revealed:
                # In a real game, this would be a choice. For now, let's say 25% chance to reveal.
                if random.random() < 0.25:
                    player.role.reveal(player)

    def run_chat_phase(self, channel_type: ChatChannelType):
        # This is a placeholder for the chat logic.
        # In a real game, this would involve LLMs generating messages.
        print(f"--- Chat Phase: {channel_type.name} ---")
        # Example of sending a message
        for player in self.players:
            if player.is_alive and channel_type in player.chat_channels:
                # Only send a message if the player is supposed to be in that channel
                if (channel_type == ChatChannelType.MAFIA and player.role.faction == Faction.MAFIA) or \
                   (channel_type == ChatChannelType.PUBLIC):
                    self.chat_manager.send_message(player, f"Hello from {player.name} in {channel_type.name}!", channel_type)

        # Example of retrieving messages
        for player in self.players:
            if player.is_alive and channel_type in player.chat_channels:
                messages = self.chat_manager.get_messages(player)
                # In a real implementation, these messages would be part of the LLM's context.
                # print(f"--- {player.name}'s view of {channel_type.name} chat ---")
                # for msg in messages:
                #     if msg.channel_type == channel_type:
                #         print(msg)

    def run_voting_phase(self):
        print("--- Voting Phase ---")
        living_players = [p for p in self.players if p.is_alive]
        
        # Placeholder voting logic: each living player votes for a random other living player.
        for player in living_players:
            targettable_players = [p for p in living_players if p != player]
            if targettable_players:
                voted_for = random.choice(targettable_players)
                player.voted_for = voted_for
                voted_for.votes_on += 1
                print(f"{player.name} voted for {voted_for.name}.")

        # Check for trial
        player_on_trial = None
        for player in living_players:
            if player.votes_on >= len(living_players) / 2:
                player_on_trial = player
                player.is_on_trial = True
                break
        
        if player_on_trial:
            self.run_trial_phase(player_on_trial)

    def run_trial_phase(self, player_on_trial):
        print(f"--- Trial Phase ---")
        print(f"{player_on_trial.name} is on trial.")
        # Placeholder trial logic: The town always votes guilty for now.
        print(f"{player_on_trial.name} has been voted guilty.")
        player_on_trial.is_alive = False
        print(f"{player_on_trial.name} has been lynched!")
        self.graveyard.append(player_on_trial)
        self.chat_manager.add_player_to_channel(player_on_trial, ChatChannelType.GRAVEYARD)
        self.chat_manager.remove_player_from_channel(player_on_trial, ChatChannelType.PUBLIC)
        if player_on_trial.role.faction == Faction.MAFIA:
            self.chat_manager.remove_player_from_channel(player_on_trial, ChatChannelType.MAFIA)

    def run_night_phase(self):
        print(f"--- Night {int(self.day)} ---")
        # Night chat for factions
        self.run_chat_phase(ChatChannelType.MAFIA)
        
        # Collect night actions from players (LLMs)
        # Process night actions based on priority
        # For now, let's simulate some simple actions.
        
        # This is a placeholder for action processing.
        # A real implementation would involve getting actions from LLMs.
        actions = self.get_placeholder_actions()
        self.process_night_actions(actions)
        
        for player in self.players:
            player.clear_night_actions()

    def get_placeholder_actions(self):
        # Placeholder: a Mafioso and SK target a random living player.
        actions = []
        godfather_player = next((p for p in self.players if p.is_alive and p.role.name == RoleName.GODFATHER), None)
        mafioso_players = [p for p in self.players if p.is_alive and p.role.name == RoleName.MAFIOSO]
        sk_players = [p for p in self.players if p.is_alive and p.role.name == RoleName.SERIAL_KILLER]
        town_investigative_players = [p for p in self.players if p.is_alive and p.role.alignment == RoleAlignment.TOWN_INVESTIGATIVE]
        town_protective_players = [p for p in self.players if p.is_alive and p.role.alignment == RoleAlignment.TOWN_PROTECTIVE]
        town_killing_players = [p for p in self.players if p.is_alive and p.role.alignment == RoleAlignment.TOWN_KILLING]
        town_support_players = [p for p in self.players if p.is_alive and p.role.alignment == RoleAlignment.TOWN_SUPPORT]
        
        living_players = [p for p in self.players if p.is_alive]

        # Mafia attack logic
        mafia_target = None
        if godfather_player:
            targettable = [p for p in living_players if p.role.faction != Faction.MAFIA]
            if targettable:
                mafia_target = random.choice(targettable)
                # Godfather orders the hit
                godfather_player.role.perform_night_action(godfather_player, mafia_target)
                # If mafioso exists, they perform the action. Otherwise, GF does.
                actor = mafioso_players[0] if mafioso_players else godfather_player
                actions.append({"actor": actor, "target": mafia_target})
        elif mafioso_players: # Handle case where only mafioso is alive
            targettable = [p for p in living_players if p.role.faction != Faction.MAFIA]
            if targettable:
                 actions.append({"actor": mafioso_players[0], "target": random.choice(targettable)})

        for sk in sk_players:
            targettable = [p for p in living_players if p != sk]
            if targettable:
                actions.append({"actor": sk, "target": random.choice(targettable)})
        
        for ti_player in town_investigative_players:
            # Spies and Psychics don't target in this simplified model yet.
            if ti_player.role.name in [RoleName.SPY, RoleName.PSYCHIC]:
                actions.append({"actor": ti_player, "target": None})
                continue
            
            targettable = [p for p in living_players if p != ti_player]
            if targettable:
                actions.append({"actor": ti_player, "target": random.choice(targettable)})

        for tp_player in town_protective_players:
            targettable = [p for p in living_players if p != tp_player]
            if targettable:
                actions.append({"actor": tp_player, "target": random.choice(targettable)})

        for tk_player in town_killing_players:
            # Vigilante can't shoot on night 1
            if tk_player.role.name == RoleName.VIGILANTE and self.day == 1:
                continue
            targettable = [p for p in living_players if p != tk_player]
            if targettable:
                actions.append({"actor": tk_player, "target": random.choice(targettable)})

        for ts_player in town_support_players:
            if ts_player.role.name == RoleName.ESCORT:
                targettable = [p for p in living_players if p != ts_player]
                if targettable:
                    actions.append({"actor": ts_player, "target": random.choice(targettable)})

        return actions

    def process_night_actions(self, actions):
        # A dictionary to track who is being attacked and by how much power
        attacks_on_players = {player.id: [] for player in self.players}

        # First, log all visits and queue up attacks
        for action in actions:
            actor = action["actor"]
            target = action["target"]

            if target:
                actor.visit(target)
                target.targeted_by.append(actor)
                if actor.role.attack.value > 0:
                    attacks_on_players[target.id].append(actor.role.attack)

        # Then, process non-attacking actions like investigations and protections
        for action in actions:
            actor = action["actor"]
            target = action["target"]

            if actor.is_alive and not actor.is_role_blocked and actor.role.attack.value == 0:
                result = actor.role.perform_night_action(actor, target)
                if result:
                    print(f"Night Action Result for {actor.name}: {result}")
        
        # Print out who the mafia and sk are targeting
        for action in actions:
            actor = action["actor"]
            target = action["target"]
            if actor.is_alive and not actor.is_role_blocked and actor.role.attack.value > 0:
                 print(f"{actor.name} ({actor.role.name.value}) is attacking {target.name}.")

        # Finally, resolve all attacks
        for player_id, attacks in attacks_on_players.items():
            if not attacks:
                continue

            player = self.players[player_id]
            if not player.is_alive:
                continue

            # Check for protection
            healed = False
            guarded = False
            for action in actions:
                if action["target"] == player:
                    # Mayor cannot be healed after reveal
                    if action["actor"].role.name == RoleName.DOCTOR and not (player.role.name == RoleName.MAYOR and player.role.revealed):
                        healed = True
                    elif action["actor"].role.name == RoleName.BODYGUARD:
                        guarded = True
            
            if healed:
                print(f"{player.name} was attacked but was saved by a Doctor!")
                continue

            # For now, we'll just take the strongest attack against the player
            strongest_attack = max(attacks, key=lambda a: a.value)

            if guarded:
                # Bodyguard saves the target but dies in the process, killing one of the attackers
                print(f"{player.name} was attacked but was saved by a Bodyguard!")
                bodyguard_player = next((p for p in self.players if p.role.name == RoleName.BODYGUARD and p.is_alive), None)
                if bodyguard_player:
                    bodyguard_player.is_alive = False
                    print(f"{bodyguard_player.name} died protecting their target!")
                    self.graveyard.append(bodyguard_player)

                    # Find an attacker to kill
                    attackers = [p for p in self.players if p.id in [a["actor"].id for a in actions if a["target"] == player and a["actor"].role.attack.value > Attack.NONE.value]]
                    if attackers:
                        attacker_to_die = attackers[0]
                        attacker_to_die.is_alive = False
                        print(f"{attacker_to_die.name} was killed by a Bodyguard!")
                        self.graveyard.append(attacker_to_die)
                continue

            if strongest_attack.value > player.defense.value:
                player.is_alive = False
                print(f"{player.name} was killed!")
                self.graveyard.append(player)
                self.chat_manager.add_player_to_channel(player, ChatChannelType.GRAVEYARD)
                self.chat_manager.remove_player_from_channel(player, ChatChannelType.PUBLIC)
                if player.role.faction == Faction.MAFIA:
                    self.chat_manager.remove_player_from_channel(player, ChatChannelType.MAFIA)
                
                # Check for Vigilante guilt
                vigilante_attacker = next((a["actor"] for a in actions if a["target"] == player and a["actor"].role.name == RoleName.VIGILANTE), None)
                if vigilante_attacker and player.role.faction == Faction.TOWN:
                    print(f"{vigilante_attacker.name} has died from guilt!")
                    vigilante_attacker.is_alive = False
                    self.graveyard.append(vigilante_attacker)

        self.check_for_promotions()

    def check_for_promotions(self):
        living_mafia = [p for p in self.players if p.is_alive and p.role.faction == Faction.MAFIA]
        godfather_present = any(p.role.name == RoleName.GODFATHER for p in living_mafia)

        if not godfather_present:
            mafiosos = [p for p in living_mafia if p.role.name == RoleName.MAFIOSO]
            if mafiosos:
                new_godfather = mafiosos[0]
                new_godfather.role = Godfather()
                print(f"{new_godfather.name} has been promoted to Godfather!")

    def check_win_conditions(self):
        # Simplified win conditions
        living_players = [p for p in self.players if p.is_alive]
        town_alive = [p for p in living_players if p.role.faction == Faction.TOWN]
        mafia_alive = [p for p in living_players if p.role.faction == Faction.MAFIA]
        neutral_killers_alive = [p for p in living_players if p.role.alignment == RoleAlignment.NEUTRAL_KILLING]

        if not mafia_alive and not neutral_killers_alive:
            self.game_over = True
            self.winner = Faction.TOWN
            print("Town wins!")
        elif len(mafia_alive) >= len(town_alive) and not neutral_killers_alive:
            self.game_over = True
            self.winner = Faction.MAFIA
            print("Mafia wins!")
        # Add more win conditions for other factions and neutrals
        elif len(living_players) <= 2 and neutral_killers_alive:
             # Assuming any NK wins if they are one of the last two standing.
             self.game_over = True
             self.winner = Faction.NEUTRAL
             print("The Serial Killer wins!") 