
"""
Comprehensive test suite for the prompt builder system.

Tests:
1. Model configuration system (Gemma vs default)
2. Tool discovery and categorization  
3. Role-specific interaction filtering
4. Observation role system (notebook, environment_static)
5. Graveyard formatting
6. Environment static tool tracking
7. Complete prompt generation for different roles
"""

import pytest
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Simulation.prompt_builder import (
    build_complete_prompt,
    get_model_config,
    discover_tools,
    get_available_interactions,
    build_role_card,
    build_notebook_observation,
    build_environment_static_observations,
    get_role_specific_interactions
)
from Simulation.player import Player
from Simulation.roles import Doctor, Sheriff, Jailor, Bodyguard, Vigilante, Veteran
from Simulation.game import Game
from Simulation.config import GameConfiguration


class TestModelConfigs:
    """Test model-specific configuration system."""
    
    def test_gemma_config(self):
        """Test Gemma model configuration."""
        config = get_model_config("gemma-7b-instruct")
        assert config.name == "gemma"
        assert config.has_system_prompt == False
        assert config.system_token == "<start_of_turn>user"
        assert config.user_token == "<start_of_turn>user"
        assert config.assistant_token == "<start_of_turn>model"
    
    def test_default_config(self):
        """Test default model configuration."""
        config = get_model_config("llama-3-8b")
        assert config.name == "default"
        assert config.has_system_prompt == True
        assert config.system_token == "<|system|>"
        assert config.user_token == "<|user|>"
    
    def test_partial_matching(self):
        """Test partial model name matching."""
        # Should match "gemma" from "gemma-2-9b"
        config = get_model_config("gemma-2-9b-it")
        assert config.name == "gemma"
        assert config.has_system_prompt == False
    
    def test_unknown_model_fallback(self):
        """Test unknown model falls back to default."""
        config = get_model_config("unknown-model-xyz")
        assert config.name == "default"
        assert config.has_system_prompt == True


class TestToolDiscovery:
    """Test tool discovery and categorization."""
    
    def test_tool_discovery(self):
        """Test that tools are discovered from JSON files."""
        tools = discover_tools()
        assert len(tools) > 0
        
        # Check that we have expected tools
        tool_names = {tool.name for tool in tools}
        expected_tools = {"view_will", "notebook", "chat_history", "graveyard"}
        assert expected_tools.issubset(tool_names)
    
    def test_tool_syntax_generation(self):
        """Test proper syntax generation for different tool types."""
        tools = discover_tools()
        tool_dict = {tool.name: tool for tool in tools}
        
        # Check syntax for different tool types
        if "view_will" in tool_dict:
            assert tool_dict["view_will"].syntax == "</view_will>"
        
        if "notebook" in tool_dict:
            assert tool_dict["notebook"].syntax == "<notebook>content</notebook>"
        
        if "chat_history" in tool_dict:
            assert tool_dict["chat_history"].syntax == "<chat_history>argument</chat_history>"
    
    def test_tool_classification(self):
        """Test tool classification (environment_static, etc.)."""
        tools = discover_tools()
        
        # Find notebook tool and verify it's environment_terminal
        notebook_tools = [t for t in tools if t.name == "notebook"]
        if notebook_tools:
            assert notebook_tools[0].tool_class == "environment_terminal"


class TestRoleSpecificInteractions:
    """Test role-specific interaction filtering."""
    
    def test_doctor_interactions(self):
        """Test Doctor gets heal ability."""
        doctor = Doctor()
        interactions = get_available_interactions(doctor)
        
        # Should have base interactions + heal
        interaction_names = {i.name for i in interactions}
        assert "speak" in interaction_names
        assert "whisper" in interaction_names
        assert "vote" in interaction_names
        assert "wait" in interaction_names
        assert "heal" in interaction_names
        
        # Check heal ability details
        heal_interaction = next(i for i in interactions if i.name == "heal")
        assert heal_interaction.syntax == "<heal>PlayerName</heal>"
        assert "Night" in heal_interaction.phases_allowed
    
    def test_sheriff_interactions(self):
        """Test Sheriff gets investigate ability."""
        sheriff = Sheriff()
        interactions = get_available_interactions(sheriff)
        
        interaction_names = {i.name for i in interactions}
        assert "investigate" in interaction_names
        
        # Sheriff should NOT have heal
        assert "heal" not in interaction_names
    
    def test_jailor_interactions(self):
        """Test Jailor gets jail and execute abilities."""
        jailor = Jailor()
        interactions = get_available_interactions(jailor)
        
        interaction_names = {i.name for i in interactions}
        assert "jail" in interaction_names
        assert "execute" in interaction_names
        
        # Check jail is day ability
        jail_interaction = next(i for i in interactions if i.name == "jail")
        assert "Day Discussion" in jail_interaction.phases_allowed
        
        # Check execute is night ability
        execute_interaction = next(i for i in interactions if i.name == "execute")
        assert "Night" in execute_interaction.phases_allowed
    
    def test_bodyguard_interactions(self):
        """Test Bodyguard gets protect ability."""
        bodyguard = Bodyguard()
        interactions = get_available_interactions(bodyguard)
        
        interaction_names = {i.name for i in interactions}
        assert "protect" in interaction_names
        
        # Should NOT have killing abilities
        assert "shoot" not in interaction_names
        assert "execute" not in interaction_names
    
    def test_vigilante_interactions(self):
        """Test Vigilante gets shoot ability."""
        vigilante = Vigilante()
        interactions = get_available_interactions(vigilante)
        
        interaction_names = {i.name for i in interactions}
        assert "shoot" in interaction_names
        
        # Should NOT have healing abilities  
        assert "heal" not in interaction_names
        assert "protect" not in interaction_names
    
    def test_veteran_interactions(self):
        """Test Veteran gets alert ability."""
        veteran = Veteran()
        interactions = get_available_interactions(veteran)
        
        interaction_names = {i.name for i in interactions}
        assert "alert" in interaction_names
        
        # Check alert details
        alert_interaction = next(i for i in interactions if i.name == "alert")
        assert alert_interaction.syntax == "<alert></alert>"


class TestObservationSystem:
    """Test observation role system."""
    
    def test_notebook_observation_empty(self):
        """Test notebook observation when empty."""
        player = Player("Alice", Doctor())
        obs = build_notebook_observation(player)
        
        assert "----------Notebook 0/1500 tokens used-------------" in obs
        assert "(Empty)" in obs
    
    def test_notebook_observation_with_content(self):
        """Test notebook observation with content."""
        player = Player("Alice", Doctor())
        player.notebook = "Player Bob seems suspicious.\nPlayer Charlie claimed Doctor."
        player.notebook_tokens = 150
        
        obs = build_notebook_observation(player)
        
        assert "----------Notebook 150/1500 tokens used-------------" in obs
        assert "Player Bob seems suspicious." in obs
        assert "Player Charlie claimed Doctor." in obs
    
    def test_environment_static_observations(self):
        """Test environment_static tool observations."""
        player = Player("Alice", Doctor())
        tools_used = ["get_role_details", "attributes"]
        
        obs = build_environment_static_observations(player, tools_used)
        
        assert "get_role_details" in obs
        assert "attributes" in obs
    
    def test_environment_static_tracking(self):
        """Test environment_static tool tracking in player."""
        player = Player("Alice", Doctor())
        
        # Initially empty
        assert hasattr(player, 'environment_static_tools_used')
        assert len(player.environment_static_tools_used) == 0
        
        # Can add tools
        player.environment_static_tools_used.add("get_role_details")
        player.environment_static_tools_used.add("attributes")
        
        assert len(player.environment_static_tools_used) == 2
        assert "get_role_details" in player.environment_static_tools_used


class TestGraveyardFormatting:
    """Test graveyard formatting."""
    
    def test_graveyard_simple_format(self):
        """Test graveyard shows simple [DEAD] format."""
        config = GameConfiguration()
        
        # Create players
        alice = Player("Alice", Doctor())
        bob = Player("Bob", Sheriff())
        players = [alice, bob]
        
        # Create game and simulate death
        game = Game(config, players)
        bob.is_alive = False
        game.graveyard = [bob]
        
        from Simulation.prompt_builder import build_user_prompt
        user_prompt = build_user_prompt(game, alice)
        
        # Should contain simple graveyard format
        assert "Bob [DEAD]" in user_prompt
        
        # Should NOT contain role details in graveyard section
        assert "Sheriff" not in user_prompt or "Bob [DEAD]" in user_prompt.split("Sheriff")[0]


class TestRoleCards:
    """Test role card building."""
    
    def test_doctor_role_card(self):
        """Test Doctor role card generation."""
        doctor = Doctor()
        card = build_role_card(doctor)
        
        assert card.name == "Doctor"
        assert card.faction == "TOWN"
        assert "heal" in str(card.active_abilities).lower()
        assert "Eliminate all threats to the Town" in card.win_condition
    
    def test_jailor_role_card(self):
        """Test Jailor role card has both day and night abilities."""
        jailor = Jailor()
        card = build_role_card(jailor)
        
        assert card.name == "Jailor"
        abilities_str = str(card.active_abilities)
        assert "jail" in abilities_str.lower()
        assert "execute" in abilities_str.lower()
        assert "DAY ABILITY" in abilities_str
        assert "NIGHT ABILITY" in abilities_str


class TestCompletePrompts:
    """Test complete prompt generation."""
    
    def test_doctor_prompt_default_model(self):
        """Test complete Doctor prompt with default model."""
        config = GameConfiguration()
        alice = Player("Alice", Doctor())
        bob = Player("Bob", Sheriff())
        players = [alice, bob]
        game = Game(config, players)
        
        prompt = build_complete_prompt(game, alice, "default")
        
        # Should contain system prompt markers
        assert "<|system|>" in prompt
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt
        
        # Should contain role info
        assert "Your name is Alice" in prompt
        assert "You are a Doctor" in prompt
        
        # Should contain heal ability
        assert "<heal>PlayerName</heal>" in prompt or "heal" in prompt.lower()
        
        # Should contain roster
        assert "Alice" in prompt
        assert "Bob" in prompt
    
    def test_doctor_prompt_gemma_model(self):
        """Test complete Doctor prompt with Gemma model."""
        config = GameConfiguration()
        alice = Player("Alice", Doctor())
        bob = Player("Bob", Sheriff())
        players = [alice, bob]
        game = Game(config, players)
        
        # Add notebook content to test observation role
        alice.notebook = "Test note"
        alice.notebook_tokens = 25
        
        prompt = build_complete_prompt(game, alice, "gemma")
        
        # Should use Gemma format
        assert "<start_of_turn>user" in prompt
        assert "<start_of_turn>model" in prompt
        assert "<|system|>" not in prompt  # No system prompt for Gemma
        
        # Should contain observation for notebook
        assert "<start_of_turn>observation" in prompt
        assert "Test note" in prompt
    
    def test_sheriff_prompt_different_from_doctor(self):
        """Test Sheriff prompt is different from Doctor."""
        config = GameConfiguration()
        alice_doctor = Player("Alice", Doctor())
        alice_sheriff = Player("Alice", Sheriff())
        bob = Player("Bob", Doctor())
        players = [alice_doctor, bob]
        game = Game(config, players)
        
        doctor_prompt = build_complete_prompt(game, alice_doctor, "default")
        
        # Change alice to sheriff  
        players[0] = alice_sheriff
        sheriff_prompt = build_complete_prompt(game, alice_sheriff, "default")
        
        # Should contain different abilities
        assert "heal" in doctor_prompt.lower()
        assert "investigate" in sheriff_prompt.lower()
        
        # Sheriff should NOT have heal ability
        heal_ability_mentions = sheriff_prompt.count("<heal>")
        assert heal_ability_mentions == 0  # Should not have heal ability syntax
    
    def test_prompt_with_graveyard(self):
        """Test prompt generation with dead players."""
        config = GameConfiguration()
        alice = Player("Alice", Doctor())
        bob = Player("Bob", Sheriff())
        charlie = Player("Charlie", Jailor())
        players = [alice, bob, charlie]
        game = Game(config, players)
        
        # Kill Bob
        bob.is_alive = False
        game.graveyard = [bob]
        
        prompt = build_complete_prompt(game, alice, "default")
        
        # Should show Bob as dead in simple format
        assert "Bob [DEAD]" in prompt
        
        # Alive roster should only have Alice and Charlie
        alive_section = prompt.split("Alive Roster")[1].split("\n\n")[0]
        assert "Alice" in alive_section
        assert "Charlie" in alive_section
        assert "Bob" not in alive_section


def test_role_specific_interactions_filtering():
    """Integration test: ensure roles only get their own abilities."""
    
    roles_and_abilities = [
        (Doctor(), ["heal"]),
        (Sheriff(), ["investigate"]), 
        (Jailor(), ["jail", "execute"]),
        (Bodyguard(), ["protect"]),
        (Vigilante(), ["shoot"]),
        (Veteran(), ["alert"])
    ]
    
    for role, expected_abilities in roles_and_abilities:
        interactions = get_available_interactions(role)
        interaction_names = {i.name for i in interactions}
        
        # Should have all expected abilities
        for ability in expected_abilities:
            assert ability in interaction_names, f"{role.name.value} missing {ability}"
        
        # Should not have other roles' abilities
        all_abilities = {"heal", "investigate", "jail", "execute", "protect", "shoot", "alert"}
        other_abilities = all_abilities - set(expected_abilities)
        
        for other_ability in other_abilities:
            assert other_ability not in interaction_names, \
                f"{role.name.value} should not have {other_ability}"


if __name__ == "__main__":
    # Run specific test if provided as argument
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        pytest.main([f"-v", f"-k", test_name, __file__])
    else:
        # Run all tests
        pytest.main([f"-v", __file__]) 