import os
import signal
import sys
import textwrap
import tempfile
import subprocess
import json
import asyncio
from typing import List, Dict, Any, Union
from datetime import datetime
import re
import time
import shlex

import anthropic
from anthropic import Anthropic
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.tree import Tree
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich.columns import Columns
from rich.rule import Rule
from rich.style import Style
from rich.text import Text
from rich.theme import Theme

class AdvancedClaudeChat:
    def __init__(self):
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.messages: List[Dict[str, Any]] = []
        self.models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
        self.current_model = "claude-3-5-sonnet-20241022"
        self.editor = os.environ.get("EDITOR", "vim")
        self.conversation_history: List[Dict[str, Any]] = []
        self.load_conversation_history()

        self.tools = [{
            "name": "bash",
            "description": "bash_20241022",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    }
                },
                "required": ["command"]
            }
        }]

        custom_theme = Theme({
            "info": "dim cyan",
            "warning": "magenta",
            "danger": "bold red",
            "user": "green",
            "assistant": "blue",
            "success": "bold green",
        })
        self.console = Console(theme=custom_theme, color_system="truecolor")
        self.update_dimensions()

    def update_dimensions(self, *args):
        self.width, self.height = os.get_terminal_size()
        self.system_prompt = f"""You are an AI assistant with shell access. The current terminal dimensions are {self.width}x{self.height}.
Adjust your responses accordingly. You can use Markdown for formatting.
You can execute shell commands using the execute_shell_command function.
Always provide the full command output to the user and explain its meaning.
"""

    async def execute_shell_command_async(self, command: str) -> str:
        """Execute a shell command asynchronously and return the output."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
            if process.returncode != 0:
                return f"Command exited with non-zero status. Stderr: {stderr.decode()}"
            return stdout.decode()
        except asyncio.TimeoutError:
            return "Command execution timed out"
        except Exception as e:
            return f"An error occurred: {str(e)}"

    async def send_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        try:
            while True:
                with self.console.status("[bold green]Claude is thinking...", spinner="dots"):
                    response = self.client.messages.create(
                        model=self.current_model,
                        max_tokens=2048,
                        system=self.system_prompt,
                        messages=self.messages,
                        tools=self.tools
                    )

                assistant_message = []
                tool_use = None

                for content_block in response.content:
                    if content_block.type == 'text':
                        assistant_message.append({"type": "text", "text": content_block.text})
                    elif content_block.type == 'tool_use':
                        tool_use = content_block
                        assistant_message.append({
                            "type": "tool_use",
                            "id": content_block.id,
                            "name": content_block.name,
                            "input": content_block.input
                        })

                if tool_use:
                    if tool_use.name == 'bash':
                        command = tool_use.input['command']
                        result = await self.execute_shell_command_async(command)
                        self.console.print(f"[bold cyan]Executed command:[/bold cyan] {command}")
                        self.console.print(f"[bold cyan]Result:[/bold cyan]\n{result}")

                        self.messages.append({"role": "assistant", "content": assistant_message})

                        tool_result = {
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": result
                            }]
                        }
                        self.messages.append(tool_result)
                        continue
                    else:
                        self.console.print(f"[bold red]Unknown tool:[/bold red] {tool_use.name}")
                        break
                else:
                    break

            self.messages.append({"role": "assistant", "content": assistant_message})
            self.print_wrapped(assistant_message, "assistant")
            await self.save_conversation()
        except Exception as e:
            self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
            self.console.print_exception()

    def print_wrapped(self, content: Union[str, List[Dict[str, Any]]], role: str):
        prefix = "You: " if role == "user" else "Claude: "

        if role == "assistant":
            if isinstance(content, list):
                for block in content:
                    if block["type"] == "text":
                        md = Markdown(block["text"])
                        panel = Panel(md, title=prefix.strip(), expand=False, border_style="bold", box=box.DOUBLE)
                        self.console.print(panel)
                    elif block["type"] == "tool_use":
                        tool_panel = Panel(
                            f"[bold]Tool:[/bold] {block['name']}\n[bold]Input:[/bold] {json.dumps(block['input'], indent=2)}",
                            title="Tool Use",
                            expand=False,
                            border_style="bold",
                            box=box.SIMPLE
                        )
                        self.console.print(tool_panel)
            else:
                md = Markdown(content)
                panel = Panel(md, title=prefix.strip(), expand=False, border_style="bold", box=box.DOUBLE)
                self.console.print(panel)
        else:
            text = Text(content)
            panel = Panel(text, title=prefix.strip(), expand=False, border_style="bold", box=box.ROUNDED)
            self.console.print(panel)

    def print_separator(self):
        self.console.rule(style="dim")

    def get_user_input(self):
        user_input = self.console.input("[bold green]You (type /help for commands):[/bold green] ").strip()
        if user_input.lower() == "/edit":
            return self.get_input_from_editor()
        return user_input

    def get_input_from_editor(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+") as tf:
            tf.write("# Type your message here. Lines starting with # will be ignored.\n")
            tf.flush()
            subprocess.call([self.editor, tf.name])
            tf.seek(0)
            content = tf.read()

        lines = [line.strip() for line in content.split('\n') if not line.strip().startswith('#')]
        return '\n'.join(lines).strip()

    def change_model(self):
        current_index = self.models.index(self.current_model)
        next_index = (current_index + 1) % len(self.models)
        self.current_model = self.models[next_index]
        self.console.print(f"[bold green]Switched to model:[/bold green] {self.current_model}")

    def print_help(self):
        help_text = """
        # Available Commands

        - `/help`: Show this help message
        - `/model`: Change the current model
        - `/edit`: Open text editor for long messages
        - `/clear`: Clear the current conversation
        - `/save`: Save the current conversation
        - `/load`: Load a previous conversation
        - `/list`: List saved conversations
        - `/quit`: Exit the chat

        ## Tips
        - Use Markdown syntax for rich formatting
        - You can ask Claude to execute shell commands
        - Be cautious with shell commands as they can modify your system
        """
        self.console.print(Panel(Markdown(help_text), title="Help", border_style="bold", expand=False))

    def clear_conversation(self):
        self.messages = []
        self.console.print("[bold green]Conversation cleared.[/bold green]")

    async def save_conversation(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/conversations/conversation_\1"
        with open(filename, "w") as f:
            json.dump(self.messages, f)
        self.console.print(f"[bold green]Conversation saved as[/bold green] {filename}")

    async def load_conversation(self, filename):
        try:
            with open(filename, "r") as f:
                self.messages = json.load(f)
            self.console.print(f"[bold green]Loaded conversation from[/bold green] {filename}")
        except FileNotFoundError:
            self.console.print(f"[bold red]File {filename} not found.[/bold red]")

    def list_conversations(self):
        conversations = [f for f in os.listdir() if f.startswith("conversation_") and f.endswith(".json")]
        if conversations:
            table = Table(title="Saved Conversations", box=box.ROUNDED)
            table.add_column("Filename", style="cyan")
            table.add_column("Date", style="magenta")
            table.add_column("Time", style="yellow")
            for conv in conversations:
                date, time = conv[12:-5].split('_')
                table.add_row(conv, date, time)
            self.console.print(table)
        else:
            self.console.print("[bold yellow]No saved conversations found.[/bold yellow]")

    def load_conversation_history(self):
        history_file = "conversation_history.json"
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    self.conversation_history = json.load(f)
            except json.JSONDecodeError:
                self.console.print("[bold yellow]Warning: Conversation history file is corrupt. Starting with an empty history.[/bold yellow]")
                self.conversation_history = []
        else:
            self.conversation_history = []

    async def save_conversation_history(self):
        history_file = "conversation_history.json"
        with open(history_file, "w") as f:
            json.dump(self.conversation_history, f)

    async def run(self):
        signal.signal(signal.SIGWINCH, self.update_dimensions)

        layout = Layout()
        layout.split(
            Layout(Panel("Welcome to Advanced Claude Chat!", style="bold"), size=3),
            Layout(Panel(Markdown("Type `/help` for available commands."), style="italic"))
        )
        self.console.print(layout)

        self.print_separator()

        while True:
            user_input = self.get_user_input()

            if user_input.lower() in ['/quit', '/exit', '/q']:
                break
            elif user_input.lower() == '/help':
                self.print_help()
            elif user_input.lower() == '/model':
                self.change_model()
            elif user_input.lower() == '/clear':
                self.clear_conversation()
            elif user_input.lower() == '/save':
                await self.save_conversation()
            elif user_input.lower().startswith('/load '):
                filename = user_input.split(maxsplit=1)[1]
                await self.load_conversation(filename)
            elif user_input.lower() == '/list':
                self.list_conversations()
            elif user_input:
                self.print_wrapped(user_input, "user")
                await self.send_message(user_input)

            self.print_separator()

        farewell_text = Text("Thank you for using Advanced Claude Chat!", style="bold blue")
        farewell_panel = Panel(farewell_text, box=box.DOUBLE)
        self.console.print(farewell_panel)
        await self.save_conversation_history()

if __name__ == "__main__":
    chat = AdvancedClaudeChat()
    asyncio.run(chat.run())
