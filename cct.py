#!/usr/bin/env python3

import argparse
import sys
import json
from typing import Dict
import os

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o"

LANGUAGES = [
    "python", "javascript", "java", "c", "cpp", "csharp", "go", "rust",
    "swift", "kotlin", "ruby", "php", "typescript", "scala", "haskell"
]

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="cct: Code Completion Tokens")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, 
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("-f", "--file", help="Read code from a file")
    parser.add_argument("-o", "--output", help="Write output to a file")
    parser.add_argument("-j", "--json", action="store_true", help="Output in JSON format")
    parser.add_argument("-v", "--verbose", action="store_true", help="Include source code in output")

    subparsers = parser.add_subparsers(dest="command", required=True)

    complete_parser = subparsers.add_parser("complete", help="Complete the given code")
    complete_parser.add_argument("code", nargs="?", help="The code to complete")

    explain_parser = subparsers.add_parser("explain", help="Explain the given code")
    explain_parser.add_argument("code", nargs="?", help="The code to explain")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze the given code")
    analyze_parser.add_argument("code", nargs="?", help="The code to analyze")

    refactor_parser = subparsers.add_parser("refactor", help="Refactor the given code")
    refactor_parser.add_argument("code", nargs="?", help="The code to refactor")

    comment_parser = subparsers.add_parser("comment", help="Add comments to the given code")
    comment_parser.add_argument("code", nargs="?", help="The code to comment")

    transpile_parser = subparsers.add_parser("transpile", help="Transpile the given code to another language")
    transpile_parser.add_argument("code", nargs="?", help="The code to transpile")
    transpile_parser.add_argument("--to", choices=LANGUAGES, required=True, help="Target language for transpilation")

    return parser.parse_args()

def get_code(args: argparse.Namespace) -> str:
    if args.file:
        with open(args.file, 'r') as f:
            return f.read()
    elif args.code:
        return args.code
    elif not sys.stdin.isatty():
        return sys.stdin.read()
    else:
        print("Error: No code provided. Use -h for help.", file=sys.stderr)
        sys.exit(1)

def process_code(client: OpenAI, args: argparse.Namespace, code: str) -> Dict:
    messages = [
        {"role": "system", "content": "You are a helpful assistant that works with code."},
        {"role": "user", "content": f"Here's the code:\n\n{code}\n\n"}
    ]

    if args.command == "complete":
        messages.append({"role": "user", "content": "Please complete this code."})
    elif args.command == "explain":
        messages.append({"role": "user", "content": "Please explain this code in detail."})
    elif args.command == "analyze":
        messages.append({"role": "user", "content": "Please analyze this code and suggest improvements."})
    elif args.command == "refactor":
        messages.append({"role": "user", "content": "Please refactor this code to improve its structure and efficiency."})
    elif args.command == "comment":
        messages.append({"role": "user", "content": "Please add detailed comments to this code."})
    elif args.command == "transpile":
        messages.append({"role": "user", "content": f"Please transpile this code to {args.to}."})

    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=0.2
    )
    return response

def format_output(response: Dict, json_output: bool, verbose: bool, source_code: str, command: str) -> str:
    if json_output:
        output = {
            "result": response.choices[0].message.content,
            "command": command,
            "model": response.model,
            "usage": response.usage.dict()
        }
        if verbose:
            output["source_code"] = source_code
        return json.dumps(output, indent=2)
    else:
        output = [
            f"Command: {command}",
            "",
            response.choices[0].message.content,
            "",
            f"Model: {response.model}",
            f"Usage: {response.usage}"
        ]
        if verbose:
            output.insert(0, f"Source code:\n{source_code}\n")
        return "\n".join(output)

def main():
    args = parse_arguments()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    code = get_code(args)
    response = process_code(client, args, code)
    
    output = format_output(response, args.json, args.verbose, code, args.command)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
    else:
        print(output)

if __name__ == "__main__":
    main()