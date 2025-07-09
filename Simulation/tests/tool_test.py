import sys
from pathlib import Path

# Add the project root directory to sys.path
# Get the ToSSim root directory (go up 3 levels from this file)
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from Simulation.roles import role_map
from Simulation.tools.registry import execute_tool, get_tool_registry

def test_tool_with_roles(tool_name: str):
    """
    Test a tool with all roles in the role_map.

    Args:
        tool_name (str): The name of the tool to test.

    Returns:
        None: Prints the results of the test for each role.
    """
    print(f"Testing tool '{tool_name}' with all roles...\n")

    for role_enum, role_cls in role_map.items():
        try:
            # Instantiate the role
            role_instance = role_cls()
            role_name = role_instance.name.value if role_instance.name else "Unknown"

            # Call the tool with the role name
            print(f"Testing role: {role_name}")
            result = execute_tool(tool_name, role_name)

            # Print the result
            print(f"Result for role '{role_name}':\n{result}\n")
        except Exception as e:
            print(f"Error testing role '{role_enum.name}': {e}\n")

    print("Testing complete.")

def test_tool_with_argument(tool_name: str, argument: str = ""):
    """
    Test a tool with a specific argument.

    Args:
        tool_name (str): The name of the tool to test.
        argument (str): The argument to pass to the tool.

    Returns:
        None: Prints the result of the test.
    """
    print(f"Testing tool '{tool_name}' with argument '{argument}'...\n")

    try:
        result = execute_tool(tool_name, argument)
        print(f"Result:\n{result}\n")
    except Exception as e:
        print(f"Error testing tool '{tool_name}' with argument '{argument}': {e}\n")

    print("Testing complete.")

def list_available_tools():
    """List all available tools in the registry."""
    try:
        registry = get_tool_registry()
        print("Available tools:")
        for tool_name, tool_spec in registry.items():
            description = tool_spec.get("description", "No description available")
            print(f"  {tool_name}: {description}")
        print()
    except Exception as e:
        print(f"Error listing tools: {e}")

def show_usage():
    """Show usage information."""
    print("Usage:")
    print("  python tool_test.py <tool_name>                    # Test tool with empty argument")
    print("  python tool_test.py <tool_name> <argument>         # Test tool with specific argument")
    print("  python tool_test.py --roles <tool_name>            # Test tool with all roles")
    print("  python tool_test.py --list                         # List available tools")
    print("  python tool_test.py --help                         # Show this help")
    print()
    print("Examples:")
    print("  python tool_test.py get_role Bodyguard            # Get info for Bodyguard role")
    print("  python tool_test.py attributes                     # Get all attributes")
    print("  python tool_test.py attributes BasicDefense       # Get specific attribute")
    print("  python tool_test.py --roles get_role              # Test get_role with all roles")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_usage()
        sys.exit(1)

    first_arg = sys.argv[1]

    # Handle special flags
    if first_arg == "--help" or first_arg == "-h":
        show_usage()
        sys.exit(0)
    
    if first_arg == "--list":
        list_available_tools()
        sys.exit(0)
    
    if first_arg == "--roles":
        if len(sys.argv) != 3:
            print("Error: --roles flag requires a tool name")
            print("Usage: python tool_test.py --roles <tool_name>")
            sys.exit(1)
        tool_name = sys.argv[2]
        test_tool_with_roles(tool_name)
        sys.exit(0)

    # Normal tool testing
    tool_name = first_arg
    argument = sys.argv[2] if len(sys.argv) > 2 else ""
    
    test_tool_with_argument(tool_name, argument)