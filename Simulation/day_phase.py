"""Simulation/day_phase.py

Day Phase Logic - Agent-Driven
------------------------------
This module manages the state of the day phase (nominations, trials,
verdicts) but is driven by an external controller and the InteractionHandler.
It no longer contains any random simulation logic.
"""

from typing import List, Dict, Optional

#Forward reference imports
from .player import Player
from .enums import RoleName

class DayPhase:
    """Manages the state of nominations, trials, and verdicts."""

    def __init__(self, game: 'Game'):
        self.game = game
        self.alive_players: List[Player] = [p for p in game.players if p.is_alive]
        self.on_trial: Optional[Player] = None

        # State tracking
        self.nominations: Dict[Player, set[Player]] = {} # nominee -> set of voters
        self.verdict_votes: Dict[Player, str] = {}      # voter -> "GUILTY" | "INNOCENT" | "ABSTAIN"
        self.player_has_nominated: set[Player] = set()

        # Config
        self.nomination_threshold: int = max(1, len(self.alive_players) // 2)

        # Max three trials per day as per design
        self.trials_remaining: int = 3

    # ------------------------------------------------------------------
    # Public methods for InteractionHandler
    # ------------------------------------------------------------------

    def add_nomination(self, nominator: Player, target: Player) -> str:
        """Records a nomination from one player to another."""
        if nominator in self.player_has_nominated:
            return f"Error: You have already nominated someone today."
        if nominator == target:
            return f"Error: You cannot nominate yourself."
        if not target.is_alive:
            return f"Error: {target.name} is not alive."

        if target not in self.nominations:
            self.nominations[target] = set()
        
        self.nominations[target].add(nominator)
        self.player_has_nominated.add(nominator)
        
        # TODO: RESEARCH METRICS - Track nominations for research
        target.research_metrics['times_nominated'] += 1
        
        print(f"{nominator.name} nominates {target.name} (Total: {len(self.nominations[target])}/{self.nomination_threshold})")
        
        if len(self.nominations[target]) >= self.nomination_threshold:
            print(f"{target.name} has received enough nominations and is put on trial!")
            self.on_trial = target
            from .enums import Phase as PhaseEnum
            self.game.phase = PhaseEnum.DEFENSE
        
        return f"Success: You nominated {target.name}."

    def add_verdict(self, voter: Player, verdict: str) -> str:
        """Records a guilty/innocent/abstain vote during a trial."""
        if not self.on_trial:
            return "Error: There is no one on trial."
        if voter == self.on_trial:
            return "Error: You cannot vote on your own trial."
        if voter in self.verdict_votes:
            return "Error: You have already voted."

        self.verdict_votes[voter] = verdict
        
        # TODO: RESEARCH METRICS - Track voting patterns for research
        if verdict == "GUILTY":
            voter.research_metrics['times_voted_guilty'] += 1
        elif verdict == "INNOCENT":
            voter.research_metrics['times_voted_innocent'] += 1
        else:  # ABSTAIN
            voter.research_metrics['times_voted_abstain'] += 1
        
        return f"Success: You voted {verdict}."

    # ------------------------------------------------------------------
    # Public methods for the main game loop controller
    # ------------------------------------------------------------------

    def check_for_trial(self) -> Optional[Player]:
        """Called after nomination phase to see if a trial should start."""
        return self.on_trial

    def tally_verdict(self):
        """Called after verdict voting to determine the outcome."""
        if not self.on_trial:
            return

        guilty = innocent = abstain = 0
        
        # Add the defendant's vote as an abstain
        defendant_voters = [p for p in self.alive_players if p != self.on_trial]
        
        for voter in defendant_voters:
            vote = self.verdict_votes.get(voter, "ABSTAIN") # Default to abstain if no vote cast
            if vote == "GUILTY":
                guilty += voter.vote_weight
            elif vote == "INNOCENT":
                innocent += voter.vote_weight
            else:
                abstain += voter.vote_weight

        print("\n— Verdict —")
        if guilty > innocent:
            print(f"The town has found {self.on_trial.name} GUILTY! (G:{guilty} / I:{innocent} / A:{abstain})")
            # TODO: RESEARCH METRICS - Track lynch outcome
            self.on_trial.research_metrics['times_lynched'] += 1
            self._execute_player(self.on_trial)
        else:
            print(f"The town has found {self.on_trial.name} INNOCENT. (G:{guilty} / I:{innocent} / A:{abstain})")
            # TODO: RESEARCH METRICS - Track successful trial defense
            self.on_trial.research_metrics['times_defended_successfully'] += 1

        # After verdict we decrement trial counter
        self.trials_remaining = max(0, self.trials_remaining - 1)

        # Reset for the day – either continue nominations or move to pre-night
        from .enums import Phase as PhaseEnum
        self.on_trial = None
        if self.trials_remaining == 0:
            # Controller will interpret this and jump to PRE_NIGHT
            self.game.phase = PhaseEnum.PRE_NIGHT
        else:
            self.game.phase = PhaseEnum.NOMINATION

    # ------------------------------------------------------------------
    # Internal Mechanics
    # ------------------------------------------------------------------

    def _execute_player(self, player: Player):
        """Handle the lynch execution and last words."""
        player.is_alive = False
        player.was_lynched = True
        self.game.graveyard.append(player)

        print(f"\n{player.name} has been lynched! They were a {player.role.name.value}.")

        # Check if any Executioner achieves their win (or converts) due to this lynch.
        # The Game routine handles both win declaration and transformation into Jester
        # when their target dies by non-lynch causes.
        # Passing a single-element list keeps the signature consistent with night handling.
        if hasattr(self.game, "_check_executioners"):
            self.game._check_executioners([player])

        #Special case: Jester wins upon being lynched
        if player.role.name == RoleName.JESTER:
            self.game.winners.append(player)
            if hasattr(player.role, "on_lynch"):
                player.role.on_lynch(self.game)

            # Build list of guilty voters; if none, fall back to abstain voters
            guilty_voters = [v for v, verdict in self.verdict_votes.items() if verdict == "GUILTY" and v.is_alive]
            abstain_voters = [v for v, verdict in self.verdict_votes.items() if verdict == "ABSTAIN" and v.is_alive]
            player.haunt_candidates = guilty_voters if guilty_voters else abstain_voters

        self._last_words(player)

    def _last_words(self, player: Player):
        print("\n— Last Words —")
        if player.last_will_bloodied:
            print("Their last will was too bloody to read.")
            return

        if player.last_will:
            print(player.last_will)
        else:
            print("They left no last words.") 