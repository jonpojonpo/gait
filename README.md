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

After installation, make the wrapper script executable:

```bash
chmod +x gait
```

Then, add the GAIT directory to your PATH or create a symlink to the `gait` script in a directory that's already in your PATH.

## Tools

GAIT includes the following tools:

1. `agt`: Sentiment Analysis Tool
2. `cchat`: Claude Chat Interface
3. `cct`: Code Completion Tool
4. `cgen`: Code Generation Tool
5. `cgt`: Claude Generate Tokens
6. `cia`: Claude Interactive Agent
7. `ochat`: OpenAI Chat Interface
8. `oge`: OpenAI Generate Embeddings
9. `ogt`: OpenAI Generate Tokens
10. `oia`: OpenAI Interactive Agent
11. `osum`: OpenAI Summarization
12. `tc`: Token Count
13. `tlt`: Translate Language Tokens

To see a list of all available tools with their descriptions, use the help command:

```bash
gait help
```

## Usage

GAIT comes with a convenient wrapper script that allows you to run any tool directly. Instead of typing `python tool_name.py`, you can now use:

```bash
gait <tool_name> [arguments]
```

For example:

### Sentiment Analysis

```bash
gait agt --sentiment "I love using this AI toolbox!"
```

### Generate Code

```bash
gait cgen --python "Write a function to calculate the Fibonacci sequence"
```

### Interactive Chat

```bash
gait ochat
```

### Token Counting

```bash
gait tc "Count the tokens in this sentence"
```

For detailed usage instructions for each tool, use the `-h` or `--help` flag:

```bash
gait <tool_name> --help
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