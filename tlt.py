#!/usr/bin/env python3

import argparse
import sys
import json
from typing import Dict
import os

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"

LANGUAGES = [
    "arabic", "bengali", "chinese", "dutch", "english", "french", "german",
    "hindi", "indonesian", "italian", "japanese", "korean", "portuguese",
    "russian", "spanish", "swahili", "swedish", "tamil", "turkish", "urdu"
]

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tlt: Translate Language Tokens")
    parser.add_argument("text", nargs="?", help="The text to translate")
    
    lang_group = parser.add_mutually_exclusive_group(required=True)
    for lang in LANGUAGES:
        lang_group.add_argument(f"--{lang}", action="store_true", help=f"Translate to {lang.capitalize()}")
    
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, 
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("-f", "--file", help="Read text from a file")
    parser.add_argument("-o", "--output", help="Write output to a file")
    parser.add_argument("-j", "--json", action="store_true", help="Output in JSON format")
    parser.add_argument("-v", "--verbose", action="store_true", help="Include source text in output")
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

def get_target_language(args: argparse.Namespace) -> str:
    for lang in LANGUAGES:
        if getattr(args, lang):
            return lang.capitalize()
    return None  # This should never happen due to required=True in add_mutually_exclusive_group

def translate_text(client: OpenAI, args: argparse.Namespace, text: str, target_lang: str) -> Dict:
    messages = [
        {"role": "system", "content": "You are a helpful assistant that translates text accurately."},
        {"role": "user", "content": f"Translate the following text to {target_lang}:\n\n{text}"}
    ]
    
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=0.3  # Lower temperature for more deterministic translations
    )
    return response

def format_output(response: Dict, json_output: bool, verbose: bool, source_text: str, target_lang: str) -> str:
    if json_output:
        output = {
            "translation": response.choices[0].message.content,
            "target_language": target_lang,
            "model": response.model,
            "usage": response.usage
        }
        if verbose:
            output["source_text"] = source_text
        return json.dumps(output, indent=2)
    else:
        output = [
            f"Translation to {target_lang}:",
            response.choices[0].message.content,
            "",
            f"Model: {response.model}",
            f"Usage: {response.usage}"
        ]
        if verbose:
            output.insert(0, f"Source text: {source_text}\n")
        return "\n".join(output)

def main():
    args = parse_arguments()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    text = get_text(args)
    target_lang = get_target_language(args)
    response = translate_text(client, args, text, target_lang)
    
    output = format_output(response, args.json, args.verbose, text, target_lang)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
    else:
        print(output)

if __name__ == "__main__":
    main()