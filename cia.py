# Define constants for file paths
LOGS_DIR = "logs/conversations"


#!/usr/bin/env python3
"""
CIA (Claude Interactive Agent) - An interactive chat interface for Anthropic's Claude model.
"""

import os
import json
import sys
import signal
import readline
from typing import List, Dict, Any, Optional
from datetime import datetime
import anthropic
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time

# Define constants for file paths
LOGS_DIR = "logs/conversations"

class ClaudeAgent:
    def __init__(self):
        self.conversation_history: List[Dict[str, Any]] = []
        self.api_key = self.get_api_key()
        self.client = anthropic.Client(self.api_key)
        self.console = Console()
        self.spinner = Spinner("dots")
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.current_response = []
        self.load_conversation_history()
        self.lock = asyncio.Lock()

        # Ensure logs directory exists
        if not os.path.exists(LOGS_DIR):
            os.makedirs(LOGS_DIR)

    def get_api_key(self) -> str:
        """Get the API key from environment variables."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY environment variable not set")
            sys.exit(1)
        return api_key

    def signal_handler(self, sig, frame):
        """Handle keyboard interrupts."""
        print("\nExiting gracefully...")
        sys.exit(0)

    def setup_signal_handler(self):
        """Set up the signal handler for keyboard interrupts."""
        signal.signal(signal.SIGINT, self.signal_handler)

    async def stream_response(self, messages: List[Dict[str, str]]) -> str:
        """Stream the response from Claude."""
        try:
            with Live(self.spinner, refresh_per_second=10) as live:
                response = ""
                async with self.client.messages.stream(
                    messages=messages,
                    model="claude-3-opus-20240229",
                    max_tokens=4096,
                ) as stream:
                    async for chunk in stream:
                        if chunk.type == "content_block_delta":
                            response += chunk.delta.text
                            if chunk.delta.text.strip():
                                live.update(Markdown(response))
                return response
        except Exception as e:
            print(f"Error streaming response: {e}")
            return ""

    def format_messages(self, user_input: str) -> List[Dict[str, str]]:
        """Format messages for the API call."""
        return [{"role": "user", "content": user_input}]

    def load_conversation_history(self):
        """Load conversation history from a JSON file."""
        try:
            history_file = os.path.join(LOGS_DIR, "conversation_history.json")
            if os.path.exists(history_file):
                with open(history_file, "r") as f:
                    self.conversation_history = json.load(f)
            else:
                self.conversation_history = []
        except Exception as e:
            print(f"Error loading conversation history: {e}")
            self.conversation_history = []

    async def save_conversation_history(self):
        """Save conversation history to a JSON file."""
        try:
            history_file = os.path.join(LOGS_DIR, "conversation_history.json")
            async with self.lock:
                with open(history_file, "w") as f:
                    json.dump(self.conversation_history, f)
        except Exception as e:
            print(f"Error saving conversation history: {e}")

    async def save_conversation(self, messages: List[Dict[str, str]]):
        """Save the current conversation to a timestamped JSON file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(LOGS_DIR, f"conversation_{timestamp}.json")
            async with self.lock:
                with open(filename, "w") as f:
                    json.dump({"messages": messages}, f)
        except Exception as e:
            print(f"Error saving conversation: {e}")

    def list_conversation_files(self):
        """List all conversation files, sorted by timestamp."""
        try:
            conversations = [f for f in os.listdir(LOGS_DIR) if f.startswith("conversation_") and f.endswith(".json")]
            return sorted([os.path.join(LOGS_DIR, f) for f in conversations])
        except Exception as e:
            print(f"Error listing conversation files: {e}")
            return []

    async def chat_loop(self):
        """Main chat loop."""
        print("Welcome to Claude Interactive Agent (CIA)!")
        print("Type 'exit' or press Ctrl+C to quit.")
        print()

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() == "exit":
                    break

                messages = self.format_messages(user_input)
                print()
                response = await self.stream_response(messages)
                print("\n")

                if response:
                    self.conversation_history.append({
                        "user": user_input,
                        "assistant": response,
                        "timestamp": datetime.now().isoformat()
                    })
                    await self.save_conversation(messages + [{"role": "assistant", "content": response}])
                    await self.save_conversation_history()

            except Exception as e:
                print(f"Error in chat loop: {e}")

async def main():
    """Main function."""
    chat = ClaudeAgent()
    chat.setup_signal_handler()
    await chat.chat_loop()

if __name__ == "__main__":
    asyncio.run(main())
