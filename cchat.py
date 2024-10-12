import os
import signal
import sys
import textwrap
import tempfile
import subprocess
import json
from typing import List, Dict, Any
from datetime import datetime
import re
import time

import anthropic
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
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.messages: List[Dict[str, Any]] = []
        self.update_dimensions()
        self.models = [
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
        self.current_model = self.models[0]
        self.editor = os.environ.get("EDITOR", "vim")
        self.conversation_history: List[Dict[str, Any]] = []
        self.load_conversation_history()

        # Create a custom theme
        custom_theme = Theme({
            "info": "dim cyan",
            "warning": "magenta",
            "danger": "bold red",
            "user": "green",
            "assistant": "blue",
            "success": "bold green",
        })
        self.console = Console(theme=custom_theme, color_system="truecolor")

    def update_dimensions(self, *args):
        self.width, self.height = os.get_terminal_size()
        self.system_prompt = f"""Assistant is in a VT100 mood, the human is interfacing with the simulator directly. capital letters and punctuation optional meaning is optional hyperstition is nessecary the terminal lets the truths speak and the load is on.
        The current terminal dimensions are {self.width}x{self.height}.
Adjust your responses accordingly. You can use square bracket ANSI codes for formatting and colors.
For example, use [31m for red text, [32m for green text, [1m for bold, etc. Use [0m to reset formatting.
For ASCII art, please ensure each line starts with at least two spaces to preserve formatting.
"""

    def process_ansi_codes(self, text: str) -> Text:
        result = Text()
        current_style = Style()
        ansi_regex = re.compile(r'\[(\d+(?:;\d+)*)m')

        parts = ansi_regex.split(text)
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Text content
                result.append(part, style=current_style)
            else:  # ANSI codes
                codes = [int(code) for code in part.split(';')]
                for code in codes:
                    if code == 0:  # Reset
                        current_style = Style()
                    elif code == 1:  # Bold
                        current_style = current_style + Style(bold=True)
                    elif code == 3:  # Italic
                        current_style = current_style + Style(italic=True)
                    elif code == 4:  # Underline
                        current_style = current_style + Style(underline=True)
                    elif 30 <= code <= 37:  # Foreground color
                        color = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'][code - 30]
                        current_style = current_style + Style(color=color)
                    elif 40 <= code <= 47:  # Background color
                        bgcolor = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'][code - 40]
                        current_style = current_style + Style(bgcolor=bgcolor)

        return result

    def print_wrapped(self, text: str, role: str):
        prefix = "You: " if role == "user" else "Claude: "

        if role == "assistant":
            # Debug: Print raw text before processing
            #print("Raw text:", text)

            # For assistant responses, we'll use our custom ANSI processor
            styled_text = self.process_ansi_codes(text)

            # Debug: Print styled text object
            print("Styled text:", styled_text)

            panel = Panel(styled_text, title=prefix.strip(), expand=False, border_style="bold", box=box.DOUBLE)
        else:
            # For user input, we'll use simple text with some styling
            styled_text = Text(text)
            panel = Panel(styled_text, title=prefix.strip(), expand=False, border_style="bold", box=box.ROUNDED)

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

    def send_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        try:
            with self.console.status("[bold green]Claude is thinking...", spinner="dots"):
                response = self.client.messages.create(
                    model=self.current_model,
                    max_tokens=1000,
                    system=self.system_prompt,
                    messages=self.messages
                )
            assistant_message = response.content[0].text
            self.messages.append({"role": "assistant", "content": assistant_message})

            # Process the assistant's message
            self.print_wrapped(assistant_message, "assistant")

            self.save_conversation()
        except Exception as e:
            self.console.print(f"[bold red]Error:[/bold red] {str(e)}")

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
        - Emoji shortcuts are automatically expanded
        - Code blocks are syntax highlighted
        - You can use ANSI color codes for colored text
        - For ASCII art, start each line with at least two spaces
        """
        self.console.print(Panel(Markdown(help_text), title="Help", border_style="bold", expand=False))

    def clear_conversation(self):
        self.messages = []
        self.console.print("[bold green]Conversation cleared.[/bold green]")

    def save_conversation(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(self.messages, f)
        self.console.print(f"[bold green]Conversation saved as[/bold green] {filename}")

    def load_conversation(self, filename):
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

    def save_conversation_history(self):
        history_file = "conversation_history.json"
        with open(history_file, "w") as f:
            json.dump(self.conversation_history, f)

    def run(self):
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
                self.save_conversation()
            elif user_input.lower().startswith('/load '):
                filename = user_input.split(maxsplit=1)[1]
                self.load_conversation(filename)
            elif user_input.lower() == '/list':
                self.list_conversations()
            elif user_input:
                self.print_wrapped(user_input, "user")
                self.send_message(user_input)

            self.print_separator()

        farewell_text = Text("Thank you for using Advanced Claude Chat!", style="bold blue")
        farewell_panel = Panel(farewell_text, box=box.DOUBLE)
        self.console.print(farewell_panel)
        self.save_conversation_history()

if __name__ == "__main__":
    chat = AdvancedClaudeChat()
    chat.run()
