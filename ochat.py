import os
import signal
import sys
import textwrap
import tempfile
import subprocess
import json
import asyncio
import time
import re
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.style import Style
from rich.text import Text
from rich.theme import Theme
from rich.prompt import Prompt

import openai

MAX_TOOL_CALLS = 128

class FunOpenAIChat:
    def __init__(self):
        # Initialize OpenAI client correctly
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.messages: List[Dict[str, Any]] = []
        self.update_dimensions()
        self.models = ["gpt-4o", "gpt-4o-mini"] 
        self.current_model = self.models[0]
        self.editor = os.environ.get("EDITOR", "vim")
        self.conversation_history: List[Dict[str, Any]] = []
        self.load_conversation_history()

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
        self.system_prompt = f"""You are a fun and quirky AI assistant in a true colour terminal mood! The human is interfacing with the simulator directly. Capital letters and punctuation are optional, meaning is optional, and hyperstition is necessary. The terminal lets the truths speak, and the load is on!

The current terminal dimensions are {self.width}x{self.height}.
Adjust your responses accordingly. You can use square bracket ANSI codes for formatting and colors.
For example, use [31m for red text, [32m for green text, [1m for bold, etc. Use [0m to reset formatting.
For ASCII art, please ensure each line starts with at least two spaces to preserve formatting.

You also have access to the following tool:
- execute_shell_command: Execute shell commands.

Be aware of the current working directory and the files you create.

Remember to be entertaining, use wordplay, and inject humor into your responses!
"""

    async def execute_shell_command_async(self, command: str) -> str:
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
            if process.returncode != 0:
                return f"[31mOops! Command threw a tantrum. Stderr says:[0m {stderr.decode()}"
            return f"[32mCommand executed successfully! Here's the output:[0m\n{stdout.decode()}"
        except asyncio.TimeoutError:
            return "[31mUh-oh! The command got lost in the void of time. It's been more than 30 seconds![0m"
        except Exception as e:
            return f"[31mWhoopsie! An error occurred: {str(e)}[0m"

    async def process_tool_calls(self, tool_calls):
        tasks = []
        for tool_call in tool_calls:
            if tool_call.function.name == "execute_shell_command":
                command = json.loads(tool_call.function.arguments)["command"]
                task = asyncio.create_task(self.execute_shell_command_async(command))
                tasks.append((tool_call, task))
        
        results = []
        for tool_call, task in tasks:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                progress_task = progress.add_task(description=f"[35mWorking some magic with {tool_call.function.name}...[0m", total=None)
                result = await task
                progress.remove_task(progress_task)
            # Replace console.print with display_response to handle ANSI codes
            self.display_response(f"[1m[33mExecuted {tool_call.function.name}:[0m [33m{tool_call.function.arguments}[0m")
            self.display_response(f"[1m[32mResult (ta-da!):[0m\n{result}")
            results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": result
            })
        return results

    async def send_message_async(self, content: str):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_shell_command",
                    "description": "Execute a shell command. Use with caution and a sprinkle of fun!",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The shell command to execute.",
                            },
                        },
                        "required": ["command"],
                    },
                }
            }
        ]

        self.messages.append({"role": "user", "content": content})
        start_time = time.time()

        try:
            response = await asyncio.to_thread(
                self.client.chat.completion.create,
                model=self.current_model,
                messages=[{"role": "system", "content": self.system_prompt}] + self.messages,
                functions=tools,
                function_call="auto",
            )
        except Exception as e:
            self.display_response(f"[1m[31mError:[0m {str(e)}")
            return

        tool_call_count = 0

        while True:
            try:
                message = response.choices[0].message

                if message.get("function_call") and tool_call_count < MAX_TOOL_CALLS:
                    # Assuming the API returns 'function_call' instead of 'tool_calls'
                    results = await self.process_tool_calls([message])
                    self.messages.append(message)  # Ensure proper message appending
                    tool_call_count += 1
                    response = await asyncio.to_thread(
                        self.client.chat.completion.create,
                        model=self.current_model,
                        messages=[{"role": "system", "content": self.system_prompt}] + self.messages,
                        functions=tools,
                        function_call="auto",
                    )
                else:
                    self.messages.append({"role": "assistant", "content": message.get("content", "")})
                    self.display_response(message.get("content", ""))
                    break

            except Exception as e:
                self.display_response(f"[1m[31mError:[0m {str(e)}")
                break

        end_time = time.time()
        self.display_response(f"[1m[34mTotal execution time: {end_time - start_time:.2f} seconds[0m (That's {(end_time - start_time) * 1000:.0f} milliseconds for the speed demons out there!)")
        self.save_conversation()

    def send_message(self, content: str):
        asyncio.run(self.send_message_async(content))

    def display_response(self, content: str):
        processed_content = self.process_ansi_codes(content)
        panel = Panel(processed_content, expand=False, border_style="bold", box=box.DOUBLE)
        self.console.print(panel)

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

    def print_separator(self):
        self.console.rule(style="dim")

    def get_user_input(self):
        prompt_text = "[bold magenta]┌──([/bold magenta][bold cyan]Human[/bold cyan][bold magenta])-[~]\n└─$[/bold magenta] "
        user_input = Prompt.ask(prompt_text)
        if user_input.lower() == "/help":
            return "/help"
        return user_input.strip()

    def change_model(self):
        current_index = self.models.index(self.current_model)
        next_index = (current_index + 1) % len(self.models)
        self.current_model = self.models[next_index]
        self.display_response(f"[1m[32mSwitched to model:[0m {self.current_model}")

    def print_help(self):
        help_text = """
        # Available Commands

        - [bold]/help[/bold]: Show this help message (you're looking at it!)
        - [bold]/model[/bold]: Change the current model (it's like changing hats, but for AIs)
        - [bold]/edit[/bold]: Open text editor for long messages (for when you're feeling extra verbose)
        - [bold}/clear[/bold]: Clear the current conversation (amnesia on demand)
        - [bold}/save[/bold]: Save the current conversation (for posterity)
        - [bold}/load[/bold]: Load a previous conversation (time travel, anyone?)
        - [bold}/list[/bold]: List saved conversations (your chat history book)
        - [bold}/quit[/bold]: Exit the chat (but why would you want to?)

        ## Tips
        - Use Markdown syntax for rich formatting (make your text pop!)
        - Code blocks are syntax highlighted (like a disco for your code)
        - You can use ANSI color codes for colored text (paint with words)
        - For ASCII art, start each line with at least two spaces (create masterpieces)
        """
        self.display_response("[1m[35m Help & Shenanigans [0m")
        self.console.print(Panel(Markdown(help_text), border_style="bold", expand=False))

    def clear_conversation(self):
        self.messages = []
        self.display_response("[1m[32mConversation cleared.[0m It's like we never met!")

    def save_conversation(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fun_conversation_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(self.messages, f)
        self.display_response(f"[1m[32mConversation saved as[0m {filename} (It's immortal now!)")

    def load_conversation(self, filename):
        try:
            with open(filename, "r") as f:
                self.messages = json.load(f)
            self.display_response(f"[1m[32mLoaded conversation from[0m {filename} (Welcome back to the future!)")
        except FileNotFoundError:
            self.display_response(f"[1m[31mFile {filename} not found.[0m (It's playing hide and seek, and winning!)")

    def list_conversations(self):
        conversations = [f for f in os.listdir() if f.startswith("fun_conversation_") and f.endswith(".json")]
        if conversations:
            self.display_response("[1m[36mSaved conversations:[0m")
            for conv in conversations:
                self.console.print(f"- {conv}")
        else:
            self.display_response("[1m[33mNo saved conversations found.[0m (It's lonely in here!)")

    def load_conversation_history(self):
        history_file = "fun_conversation_history.json"
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                self.messages = json.load(f)

    def save_conversation_history(self):
        history_file = "fun_conversation_history.json"
        with open(history_file, "w") as f:
            json.dump(self.messages, f)

    def run(self):
        signal.signal(signal.SIGWINCH, self.update_dimensions)
        self.display_response("[1m[35m Welcome to the Fun OpenAI Chat! Where AI meets witty banter! [0m")
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
            elif user_input.lower() == '/edit':
                user_input = self.get_input_from_editor()
                if user_input:
                    self.display_response(f"[bold cyan]You:[/bold cyan] {user_input}")
                    self.print_separator()
                    self.send_message(user_input)
                    self.print_separator()
            elif user_input:
                self.display_response(f"[bold cyan]You:[/bold cyan] {user_input}")
                self.print_separator()
                self.send_message(user_input)
                self.print_separator()

        self.display_response("[1m[35mThanks for chatting! May your code be bug-free and your puns be groan-worthy![0m")
        self.save_conversation_history()

    def get_input_from_editor(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+") as tf:
            tf.write("# Type your message here. Lines starting with # will be ignored.\n")
            tf.flush()
            subprocess.call([self.editor, tf.name])
            tf.seek(0)
            content = tf.read()
            # Remove lines starting with #
            lines = content.splitlines()
            cleaned_lines = [line for line in lines if not line.strip().startswith('#')]
            return '\n'.join(cleaned_lines).strip()

if __name__ == '__main__':
    chat = FunOpenAIChat()
    try:
        chat.run()
    except KeyboardInterrupt:
        chat.display_response("\n[1m[35mInterrupted! Exiting the Fun OpenAI Chat. See you next time! [0m")
        chat.save_conversation_history()
        sys.exit(0)
