#!/usr/bin/env python3

import argparse
import sys
import json
from typing import List, Dict

import tiktoken
from tiktoken._educational import SimpleBytePairEncoding

DEFAULT_MODEL = "o200k_base"  # Default tokenizer for GPT-4o and GPT-4o-mini

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tc: Token Count")
    parser.add_argument("text", nargs="?", help="The text to tokenize")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, 
                        help=f"Tokenizer model to use (default: {DEFAULT_MODEL}, used by GPT-4o and GPT-4o-mini)")
    parser.add_argument("-c", "--count", action="store_true", help="Only output the token count")
    parser.add_argument("-t", "--tokens", action="store_true", help="Output the individual tokens")
    parser.add_argument("-e", "--educational", action="store_true", help="Use educational mode to visualize tokenization")
    parser.add_argument("-p", "--pretty", action="store_true", help="Pretty print the tokenization (use with -e)")
    parser.add_argument("-f", "--file", help="Read text from a file")
    parser.add_argument("-o", "--output", help="Write output to a file")
    parser.add_argument("-j", "--json", action="store_true", help="Output in JSON format")
    return parser.parse_args()

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

def tokenize(text: str, model: str, educational: bool = False) -> List[int]:
    if educational:
        enc = SimpleBytePairEncoding.from_tiktoken(model)
    else:
        enc = tiktoken.get_encoding(model)
    return enc.encode(text)

def pretty_print_tokens(text: str, tokens: List[int], enc: SimpleBytePairEncoding) -> str:
    output = []
    for token in tokens:
        decoded = enc.decode([token])
        output.append(f"{token}: {decoded!r}")
    return "\n".join(output)

def main():
    args = parse_arguments()
    text = get_text(args)
    
    if args.educational:
        enc = SimpleBytePairEncoding.from_tiktoken(args.model)
        tokens = enc.encode(text)
    else:
        enc = tiktoken.get_encoding(args.model)
        tokens = enc.encode(text)
    
    token_count = len(tokens)
    
    if args.json:
        result = {
            "count": token_count,
            "tokens": tokens if args.tokens else None,
            "pretty": pretty_print_tokens(text, tokens, enc) if args.pretty and args.educational else None
        }
        output = json.dumps(result, indent=2)
    elif args.count:
        output = str(token_count)
    elif args.tokens:
        output = " ".join(map(str, tokens))
    elif args.educational and args.pretty:
        output = pretty_print_tokens(text, tokens, enc)
    else:
        output = f"Token count: {token_count}"
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
    else:
        print(output)

if __name__ == "__main__":
    main()