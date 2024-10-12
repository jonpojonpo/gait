# GAIT (Gen AI Toolbox)

GAIT is a comprehensive collection of command-line tools designed to harness the power of OpenAI's language models and Anthropic's Claude for various natural language processing tasks. This toolbox provides a set of utilities that make it easy to interact with AI models, perform text analysis, and leverage AI capabilities in your workflows.

## Table of Contents

- [Installation](#installation)
- [Tools](#tools)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

To use GAIT, you'll need Python 3.7 or later. Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/gait.git
cd gait
pip install -r requirements.txt
```

Make sure to set up your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

For tools that use Claude, set up your Anthropic API key:

```bash
export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
```

## Tools

GAIT includes the following tools:

1. `agt.py`: Sentiment Analysis Tool
2. `cchat.py`: Claude Chat Interface
3. `cct.py`: Code Completion Tool
4. `cgen.py`: Code Generation Tool
5. `cgt.py`: Claude Generate Tokens
6. `cia.py`: Claude Interactive Agent
7. `ochat.py`: OpenAI Chat Interface
8. `oge.py`: OpenAI Generate Embeddings
9. `ogt.py`: OpenAI Generate Tokens
10. `oia.py`: OpenAI Interactive Agent
11. `osum.py`: OpenAI Summarization
12. `tc.py`: Token Count
13. `tlt.py`: Translate Language Tokens

## Usage

Each tool in the GAIT collection can be run from the command line. Here are some examples:

### Sentiment Analysis

```bash
python agt.py --sentiment "I love using this AI toolbox!"
```

### Generate Code

```bash
python cgen.py --python "Write a function to calculate the Fibonacci sequence"
```

### Interactive Chat

```bash
python ochat.py
```

### Token Counting

```bash
python tc.py "Count the tokens in this sentence"
```

For detailed usage instructions for each tool, use the `-h` or `--help` flag:

```bash
python <tool_name>.py --help
```

## Contributing

Contributions to GAIT are welcome! Please feel free to submit pull requests, report bugs, or suggest new features.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.