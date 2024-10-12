#!/usr/bin/env python3

import argparse
import sys
import json
from typing import Dict
import os

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o"

ANALYSIS_TYPES = [
    "sentiment", "emotion", "subjectivity", "sarcasm", "stance",
    "intent", "topic", "keywords", "entities", "summary"
]

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="agt: Sentiment Analysis Tool")
    parser.add_argument("text", nargs="*", help="Text to analyze")
    
    analysis_group = parser.add_mutually_exclusive_group(required=True)
    for analysis in ANALYSIS_TYPES:
        analysis_group.add_argument(f"--{analysis}", action="store_true", help=f"Perform {analysis} analysis")
    
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("-f", "--file", help="Read text from a file")
    parser.add_argument("-o", "--output", help="Write output to a file")
    parser.add_argument("-j", "--json", action="store_true", help="Output in JSON format")
    parser.add_argument("-v", "--verbose", action="store_true", help="Include original text in output")
    return parser.parse_args()

def get_text(args: argparse.Namespace) -> str:
    if args.file:
        with open(args.file, 'r') as f:
            return f.read().strip()
    elif args.text:
        return ' '.join(args.text)
    elif not sys.stdin.isatty():
        return sys.stdin.read().strip()
    else:
        print("Error: No text provided. Use -h for help.", file=sys.stderr)
        sys.exit(1)

def get_analysis_type(args: argparse.Namespace) -> str:
    for analysis in ANALYSIS_TYPES:
        if getattr(args, analysis):
            return analysis
    return None  # This should never happen due to required=True in add_mutually_exclusive_group

def process_request(client: OpenAI, args: argparse.Namespace, text: str, analysis_type: str) -> Dict:
    messages = [
        {"role": "system", "content": f"You are an expert in {analysis_type} analysis. Provide a concise, accurate {analysis_type} analysis of the given text."},
        {"role": "user", "content": f"Analyze the following text:\n\n{text}"}
    ]

    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=0.2
    )
    return response

def format_output(response: Dict, json_output: bool, verbose: bool, text: str, analysis_type: str) -> str:
    if json_output:
        output = {
            "analysis": response.choices[0].message.content,
            "analysis_type": analysis_type,
            "model": response.model,
            "usage": response.usage.dict()
        }
        if verbose:
            output["original_text"] = text
        return json.dumps(output, indent=2)
    else:
        output = [
            f"{analysis_type.capitalize()} Analysis:",
            response.choices[0].message.content,
            "",
            f"Model: {response.model}",
            f"Usage: {response.usage}"
        ]
        if verbose:
            output.insert(0, f"Original text: {text}\n")
        return "\n".join(output)

def main():
    args = parse_arguments()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    text = get_text(args)
    analysis_type = get_analysis_type(args)
    response = process_request(client, args, text, analysis_type)
    
    output = format_output(response, args.json, args.verbose, text, analysis_type)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
    else:
        print(output)

if __name__ == "__main__":
    main()