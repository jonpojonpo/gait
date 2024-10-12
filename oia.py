#!/usr/bin/env python3
import os
import signal
import sys
import textwrap
import tempfile
import subprocess
import json
import shlex
import asyncio
import time
import re
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# Import rich library components
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.panel import Panel
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn

import openai

MAX_TOOL_CALLS = 128  # Maximum number of tool calls allowed

class ImprovedOpenAIChat:
    def __init__(self):
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.messages: List[Dict[str, Any]] = []
        self.models = ["gpt-4o-mini","gpt-4o", "gpt-4-turbo"]
        self.current_model = self.models[0]
        self.editor = os.environ.get("EDITOR", "vim")
        self.conversation_history: List[Dict[str, Any]] = []
        self.load_conversation_history()
        self.update_dimensions()

        # Initialize Rich console
        self.console = Console()
        self.script_dir = Path("./scripts")
        self.script_dir.mkdir(exist_ok=True)

        # Display welcome message
        self.display_welcome_message()

    def display_welcome_message(self):
        from pyfiglet import Figlet
        fig = Figlet(font='slant')
        welcome_text = fig.renderText('OpenAI Chat')
        self.console.print(f"[bold cyan]{welcome_text}[/bold cyan]")
        self.console.print("[bold green]Welcome to the Improved OpenAI Chat with Function Calling![/bold green]")
        self.console.print("Type [bold]/help[/bold] for available commands.")
        self.print_separator()

    def update_dimensions(self, *args):
        self.width, self.height = os.get_terminal_size()
        self.system_prompt = f"""You are an AI assistant. The current terminal dimensions are {self.width}x{self.height}. 
You can use colors and Unicode in responses. You also have access to the following tools:
- execute_shell_command: Execute shell commands.
- save_and_run_code: Save code to a file and optionally execute it.

Be aware of the current working directory and the files you create in ./scripts/.
"""

    async def execute_shell_command_async(self, command: str) -> str:
        """Execute a shell command asynchronously and return the output."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
            if process.returncode != 0:
                return f"Command exited with non-zero status. Stderr: {stderr.decode()}"
            return stdout.decode()
        except asyncio.TimeoutError:
            return "Command execution timed out"
        except Exception as e:
            return f"An error occurred: {str(e)}"

    async def save_and_run_code_async(self, code: str, language: str, execute: bool) -> str:
        """Save code to a file and optionally execute it."""
        try:
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_script"
            extension = {
                "python": ".py",
                "javascript": ".js",
                "bash": ".sh",
                "c": ".c",
                "cpp": ".cpp",
                "java": ".java",
                "go": ".go",
                "ruby": ".rb",
                "perl": ".pl",
                "php": ".php",
                "rust": ".rs",
            }.get(language.lower(), ".txt")
            filepath = self.script_dir / f"{filename}{extension}"
            filepath.write_text(code)
            result = f"Code saved to {filepath}"
            if execute:
                if extension == ".py":
                    command = f"python3 {filepath}"
                elif extension == ".sh":
                   command = f"bash {filepath}"
                elif extension == ".go":
                    command = f"go run {filepath}"
                elif extension == ".rb":
                    command = f"ruby {filepath}"
                else:
                    return f"Execution not supported for language: {language}"
                exec_result = await self.execute_shell_command_async(command)
                result += f"\nExecution result:\n{exec_result}"
            return result
        except Exception as e:
            return f"An error occurred: {str(e)}"

    async def process_tool_calls(self, tool_calls):
        tasks = []
        for tool_call in tool_calls:
            if tool_call.function.name == "execute_shell_command":
                command = json.loads(tool_call.function.arguments)["command"]
                task = asyncio.create_task(self.execute_shell_command_async(command))
                tasks.append((tool_call, task))
            elif tool_call.function.name == "save_and_run_code":
                args = json.loads(tool_call.function.arguments)
                code = args["code"]
                language = args.get("language", "text")
                execute = args.get("execute", False)
                task = asyncio.create_task(self.save_and_run_code_async(code, language, execute))
                tasks.append((tool_call, task))
        
        results = []
        for tool_call, task in tasks:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                progress.add_task(description=f"Processing {tool_call.function.name}...", total=None)
                result = await task
            self.console.print(f"[bold yellow]Executed {tool_call.function.name}:[/bold yellow] {tool_call.function.arguments}")
            self.console.print(f"[bold green]Result:[/bold green]\n{result}")
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
                    "description": "Execute a shell command. Use with caution.",
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
            },
            {
                "type": "function",
                "function": {
                    "name": "save_and_run_code",
                    "description": "Save code to a file and optionally execute it.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The code to save.",
                            },
                            "language": {
                                "type": "string",
                                "description": "The programming language of the code.",
                            },
                            "execute": {
                                "type": "boolean",
                                "description": "Whether to execute the code after saving.",
                            },
                        },
                        "required": ["code"],
                    },
                }
            }
        ]

        self.messages.append({"role": "user", "content": content})
        start_time = time.time()

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.current_model,
                messages=[{"role": "system", "content": self.system_prompt}] + self.messages,
                tools=tools,
                tool_choice="auto",
            )
        except Exception as e:
            self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
            return

        tool_call_count = 0

        while True:
            try:
                message = response.choices[0].message

                if message.tool_calls and tool_call_count < MAX_TOOL_CALLS:
                    results = await self.process_tool_calls(message.tool_calls)
                    self.messages.append(message.model_dump())
                    self.messages.extend(results)
                    tool_call_count += 1
                    response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model=self.current_model,
                        messages=[{"role": "system", "content": self.system_prompt}] + self.messages,
                        tools=tools,
                        tool_choice="auto",
                    )
                else:
                    self.messages.append({"role": "assistant", "content": message.content})
                    self.display_response(message.content)
                    break

            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
                break

        end_time = time.time()
        self.console.print(f"[bold blue]Total execution time: {end_time - start_time:.2f} seconds[/bold blue]")

        # Save conversation after each interaction
        self.save_conversation()

    def send_message(self, content: str):
        asyncio.run(self.send_message_async(content))

    def display_response(self, content: str):
        if "```" in content:
            # Render Markdown with syntax highlighting
            md = Markdown(content)
            self.console.print(md)
        else:
            self.console.print(content)

    def print_separator(self):
        self.console.print("-" * self.width)

    def get_user_input(self):
        prompt_text = "[bold magenta]┌──([/bold magenta][bold cyan]User[/bold cyan][bold magenta])-[~]\n└─$[/bold magenta] "
        user_input = Prompt.ask(prompt_text)
        if user_input.strip() == "/help":
            return "/help"
        return user_input.strip()

    def change_model(self):
        current_index = self.models.index(self.current_model)
        next_index = (current_index + 1) % len(self.models)
        self.current_model = self.models[next_index]
        self.console.print(f"[bold green]Switched to model:[/bold green] {self.current_model}")

    def print_help(self):
        help_text = """
        [bold cyan]Commands:[/bold cyan]
        /help - Show this help message
        /model - Change the current model
        /edit - Open text editor for long messages
        /clear - Clear the current conversation
        /save - Save the current conversation
        /load <filename> - Load a previous conversation
        /list - List saved conversations
        /quit - Exit the chat
        """
        self.console.print(textwrap.dedent(help_text))

    def clear_conversation(self):
        self.messages = []
        self.console.print("[bold green]Conversation cleared.[/bold green]")

    def save_conversation(self):
        filename = "conversation.json"
        with open(filename, "w") as f:
            json.dump(self.messages, f)
        self.console.print(f"[bold green]Conversation saved to {filename}[/bold green]")

    def load_conversation(self, filename):
        try:
            with open(filename, "r") as f:
                self.messages = json.load(f)
            self.console.print(f"[bold green]Loaded conversation from {filename}[/bold green]")
        except FileNotFoundError:
            self.console.print(f"[bold red]File {filename} not found.[/bold red]")

    def list_conversations(self):
        conversations = [f for f in os.listdir() if f.startswith("conversation") and f.endswith(".json")]
        if conversations:
            self.console.print("[bold cyan]Saved conversations:[/bold cyan]")
            for conv in conversations:
                self.console.print(f"- {conv}")
        else:
            self.console.print("[bold yellow]No saved conversations found.[/bold yellow]")

    def load_conversation_history(self):
        history_file = "conversation.json"
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                self.messages = json.load(f)

    def save_conversation_history(self):
        history_file = "conversation.json"
        with open(history_file, "w") as f:
            json.dump(self.messages, f)

    def run(self):
        signal.signal(signal.SIGWINCH, self.update_dimensions)
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
                    self.console.print(f"[bold cyan]You:[/bold cyan] {user_input}")
                    self.print_separator()
                    self.send_message(user_input)
                    self.print_separator()
            elif user_input:
                self.console.print(f"[bold cyan]You:[/bold cyan] {user_input}")
                self.print_separator()
                self.send_message(user_input)
                self.print_separator()

        self.console.print("[bold green]Thank you for using Improved OpenAI Interactive Agent Chat with Function Calling![/bold green]")
        self.save_conversation_history()

    def get_input_from_editor(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+") as tf:
            tf.write("# Type your message here. Lines starting with # will be ignored.\n")
            tf.flush()
            subprocess.call([self.editor, tf.name])
            tf.seek(0)
            content = tf.read()

        lines = [line.strip() for line in content.split('\n') if not line.strip().startswith('#')]
        return '\n'.join(lines).strip()

if __name__ == "__main__":
    chat = ImprovedOpenAIChat()
    chat.run()