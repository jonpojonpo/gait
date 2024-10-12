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
    parser = argparse.ArgumentParser(description="cct: Code Creation Tool")
    parser.add_argument("description", nargs="*", help="Description of the code you want to create")
    
    lang_group = parser.add_mutually_exclusive_group()
    for lang in LANGUAGES:
        lang_group.add_argument(f"--{lang}", action="store_true", help=f"Generate code in {lang.capitalize()}")
    
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("-f", "--file", help="Read description from a file")
    parser.add_argument("-o", "--output", help="Write output to a file")
    parser.add_argument("-j", "--json", action="store_true", help="Output in JSON format")
    parser.add_argument("-v", "--verbose", action="store_true", help="Include description in output")
    return parser.parse_args()

def get_description(args: argparse.Namespace) -> str:
    if args.file:
        with open(args.file, 'r') as f:
            return f.read().strip()
    elif args.description:
        return ' '.join(args.description)
    elif not sys.stdin.isatty():
        return sys.stdin.read().strip()
    else:
        print("Error: No description provided. Use -h for help.", file=sys.stderr)
        sys.exit(1)

def get_target_language(args: argparse.Namespace) -> str:
    for lang in LANGUAGES:
        if getattr(args, lang):
            return lang.capitalize()
    return None

def process_request(client: OpenAI, args: argparse.Namespace, description: str, target_lang: str) -> Dict:
    messages = [
        {"role": "system", "content": "You are an expert programmer capable of creating code based on descriptions."},
        {"role": "user", "content": f"Create code for the following description:\n\n{description}"}
    ]

    if target_lang:
        messages[1]["content"] += f"\n\nPlease write the code in {target_lang}."

    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=0.2
    )
    return response

def format_output(response: Dict, json_output: bool, verbose: bool, description: str, lang: str) -> str:
    if json_output:
        output = {
            "code": response.choices[0].message.content,
            "language": lang or "unspecified",
            "model": response.model,
            "usage": response.usage.dict()
        }
        if verbose:
            output["description"] = description
        return json.dumps(output, indent=2)
    else:
        output = [
            response.choices[0].message.content,
            "",
            f"Language: {lang or 'unspecified'}",
            f"Model: {response.model}",
            f"Usage: {response.usage}"
        ]
        if verbose:
            output.insert(0, f"Description: {description}\n")
        return "\n".join(output)

def main():
    args = parse_arguments()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    description = get_description(args)
    target_lang = get_target_language(args)
    response = process_request(client, args, description, target_lang)
    
    output = format_output(response, args.json, args.verbose, description, target_lang)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
    else:
        print(output)

if __name__ == "__main__":
    main()