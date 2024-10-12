#!/usr/bin/env python3

import argparse
import sys
import json
from typing import List, Dict, Union
import os

from openai import OpenAI

DEFAULT_MODEL = "text-embedding-3-large"
DEFAULT_ENCODING_FORMAT = "float"

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="oge: OpenAI Generate Embeddings")
    parser.add_argument("text", nargs="?", help="The text to generate embeddings for")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, 
                        help=f"Embedding model to use (default: {DEFAULT_MODEL}) e.g text-embedding-3-small")
    parser.add_argument("-f", "--file", help="Read text from a file")
    parser.add_argument("-o", "--output", help="Write output to a file")
    parser.add_argument("-j", "--json", action="store_true", help="Output in JSON format")
    parser.add_argument("-e", "--encoding", choices=["float", "base64"], default=DEFAULT_ENCODING_FORMAT,
                        help=f"Encoding format for embeddings (default: {DEFAULT_ENCODING_FORMAT})")
    parser.add_argument("-d", "--dimensions", type=int, help="Number of dimensions for the embedding (only for text-embedding-3 and later)")
    parser.add_argument("-u", "--user", help="Unique identifier representing your end-user")
    return parser.parse_args()

def get_text(args: argparse.Namespace) -> Union[str, List[str]]:
    if args.file:
        with open(args.file, 'r') as f:
            return f.read().splitlines()  # Return a list of lines
    elif args.text:
        return args.text
    elif not sys.stdin.isatty():
        return sys.stdin.read().splitlines()  # Return a list of lines from stdin
    else:
        print("Error: No text provided. Use -h for help.", file=sys.stderr)
        sys.exit(1)

def generate_embeddings(client: OpenAI, args: argparse.Namespace, text: Union[str, List[str]]) -> Dict:
    params = {
        "model": args.model,
        "input": text,
        "encoding_format": args.encoding
    }
    if args.dimensions:
        params["dimensions"] = args.dimensions
    if args.user:
        params["user"] = args.user

    response = client.embeddings.create(**params)
    return response

def format_output(response: Dict, json_output: bool) -> str:
    if json_output:
        return json.dumps(response, indent=2)
    else:
        output = []
        for embedding in response.data:
            output.append(f"Embedding {embedding.index}:")
            output.append(str(embedding.embedding))
            output.append("")  # Empty line for readability
        output.append(f"Model: {response.model}")
        output.append(f"Usage: {response.usage}")
        return "\n".join(output)

def main():
    args = parse_arguments()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    text = get_text(args)
    response = generate_embeddings(client, args, text)
    
    output = format_output(response, args.json)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
    else:
        print(output)

if __name__ == "__main__":
    main()