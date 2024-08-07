# Token Counter

Token Counter is a command-line tool for counting tokens in text using various encoders, primarily designed for use with
language models like GPT.

## Features

- Count tokens in text files or from standard input
- Support for multiple input files
- Use different encoders, including model-specific encoders
- List available encoders
- Verbose and quiet modes for adjustable output
- Option to write results to an output file

## Installation

To install Token Counter, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/token-counter.git
   cd token-counter
   ```

2. Install the package:
   ```
   pip install .
   ```

## Usage

After installation, you can use the `token-counter` command:

```
token-counter [-h] (-t TEXT [TEXT ...] | -l) [-m MODEL] [-e ENCODER] [-o OUTPUT] [-v | -q]
```

Arguments:

- `-t`, `--text`: Text file(s) to count tokens from
- `-l`, `--list`: List available encoders
- `-m`, `--model`: Specify an encoder based on the OpenAI model name
- `-e`, `--encoder`: Token encoder to use (default: cl100k_base)
- `-o`, `--output`: Output file to write results
- `-v`, `--verbose`: Increase output verbosity
- `-q`, `--quiet`: Decrease output verbosity

## Examples

1. Count tokens in a file:
   ```
   token-counter -t sample.txt
   ```

2. Count tokens in multiple files:
   ```
   token-counter -t file1.txt file2.txt file3.txt
   ```

3. Use a specific model's encoder:
   ```
   token-counter -t sample.txt -m gpt-3.5-turbo
   ```

4. List available encoders:
   ```
   token-counter -l
   ```

5. Write results to an output file:
   ```
   token-counter -t sample.txt -o results.txt
   ```

## Development

To set up the development environment:

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install development dependencies:
   ```
   pip install -e ".[dev]"
   ```

To run tests:

```
python -m unittest discover tests
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.