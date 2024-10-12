#!/usr/bin/env python3

import argparse
import os
import sys
from typing import List, Dict, Any
import json

import anthropic

DEFAULT_MODEL = "claude-3-5-sonnet-20240620"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 1.0

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="cgt: Claude Generate Tokens")
    parser.add_argument("prompt", nargs="?", help="The prompt to send to Claude")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Claude model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("-n", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"Maximum number of tokens in the response (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("-T", "--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("-s", "--stream", action="store_true", help="Stream the output token by token")
    parser.add_argument("-f", "--file", help="Read prompt from a file")
    parser.add_argument("-o", "--output", help="Write response to a file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output (include model name, token count, etc.)")
    parser.add_argument("-S", "--system", help="Set a custom system message")
    parser.add_argument("-j", "--json", action="store_true", help="Output response in JSON format")
    return parser.parse_args()

def get_prompt(args: argparse.Namespace) -> str:
    if args.file:
        with open(args.file, 'r') as f:
            return f.read().strip()
    elif args.prompt:
        return args.prompt
    elif not sys.stdin.isatty():
        return sys.stdin.read().strip()
    else:
        print("Error: No prompt provided. Use -h for help.", file=sys.stderr)
        sys.exit(1)

def create_messages(prompt: str, system_message: str = None) -> List[Dict[str, str]]:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    return messages

def generate_response(client: anthropic.Anthropic, args: argparse.Namespace, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    stream = client.messages.create(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        messages=messages,
        stream=args.stream
    )

    if args.stream:
        response_text = ""
        for chunk in stream:
            if chunk.content:
                content = chunk.content[0].text
                print(content, end="", flush=True)
                response_text += content
        print()  # New line after streaming
        return {"content": [{"text": response_text}]}
    else:
        response = stream
        return {
            "content": response.content,
            "model": response.model,
            "role": response.role,
            "usage": response.usage
        }

def main():
    args = parse_arguments()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = get_prompt(args)
    messages = create_messages(prompt, args.system)

    response = generate_response(client, args, messages)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(response["content"][0]["text"])

    if args.verbose:
        print(f"Model: {args.model}", file=sys.stderr)
        print(f"Max tokens: {args.max_tokens}", file=sys.stderr)
        print(f"Temperature: {args.temperature}", file=sys.stderr)
        if "usage" in response:
            print(f"Input tokens: {response['usage'].input_tokens}", file=sys.stderr)
            print(f"Output tokens: {response['usage'].output_tokens}", file=sys.stderr)

    if args.json:
        print(json.dumps(response))
    elif not args.stream and not args.output:
        print(response["content"][0]["text"])

if __name__ == "__main__":
    main()