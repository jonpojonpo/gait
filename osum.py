#!/usr/bin/env python3

import argparse
import sys
import json
from typing import List, Dict
import os

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 150

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="osum: OpenAI Summarization")
    parser.add_argument("text", nargs="?", help="The text to summarize")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, 
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("-n", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Maximum number of tokens in the summary (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("-t", "--temperature", type=float, default=0.7,
                        help="Temperature for sampling (default: 0.7)")
    parser.add_argument("-f", "--file", help="Read text from a file")
    parser.add_argument("-o", "--output", help="Write output to a file")
    parser.add_argument("-j", "--json", action="store_true", help="Output in JSON format")
    return parser.parse_arguments()

def get_text(args: argparse.Namespace) -> str:
    if args.file:
        with open(args.file, 'r') as f:
            return f.read()
    elif args.text:
        return args.text
    elif not sys.stdin.isatty():
        return sys.stdin.read()
    else:
        print("Error: No text provided. Use -h for help.", file=sys.stderr)
        sys.exit(1)

def summarize_text(client: OpenAI, args: argparse.Namespace, text: str) -> Dict:
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Please summarize the following text:\n\n{text}"}
        ],
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    return response

def format_output(response: Dict, json_output: bool) -> str:
    if json_output:
        return json.dumps(response, indent=2)
    else:
        summary = response.choices[0].message.content
        output = [
            "Summary:",
            summary,
            "",
            f"Model: {response.model}",
            f"Usage: {response.usage}"
        ]
        return "\n".join(output)

def main():
    args = parse_arguments()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    text = get_text(args)
    response = summarize_text(client, args, text)
    
    output = format_output(response, args.json)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
    else:
        print(output)

if __name__ == "__main__":
    main()