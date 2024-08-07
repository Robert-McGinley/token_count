import argparse
import datetime
import fileinput
import logging
import sys
from typing import Optional

import tiktoken

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_encoders() -> list[str]:
    """Return a list of available encoders."""
    return tiktoken.list_encoding_names()


class EncodingNotFoundError(Exception):
    """Custom exception for when an encoding is not found."""
    pass


def get_encoder_by_model(model: str) -> str:
    """
    Get the encoder name for a given model.

    :param model: The model name to look up.
    :return: The encoder name for the given model.
    :raises EncodingNotFoundError: If the encoder for the model is not found.
    """
    try:
        ret = tiktoken.encoding_for_model(model).name
    except KeyError as e:
        raise EncodingNotFoundError(f"Model to encoder name lookup failed for {model}: {e}")

    if not ret:
        raise EncodingNotFoundError(f"Model to encoder name lookup failed for {model}: Empty result")
    return ret


def count_tokens(text: str, encoder: str = 'cl100k_base') -> Optional[int]:
    """
    Count tokens in a given text.

    :param text: Text to count tokens from.
    :param encoding: Encoding of the input text (default: utf-8).
    :param encoder: Encoder to use for tokenization (default: cl100k_base).
    :return: Number of tokens in the text, or None if an error occurs.
    """
    logger.debug(f"Counting tokens with encoder: {encoder}")
    try:
        enc = tiktoken.get_encoding(encoder)
    except KeyError as e:
        logger.error(f"Error creating encoder '{encoder}': {e}")
        return None

    try:
        tokens = enc.encode(text)
        return len(tokens)
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        return None


def process_input(files: Optional[list[str]] = None) -> str:
    """
    Process input files or stdin and return the combined text.

    :param files: List of file paths to process. If None, reads from stdin.
    :return: Combined text from all inputs.
    """
    input_text = ''
    for line in fileinput.input(files=files if files else ('-',)):
        input_text += line
    return input_text


def benchmark_encoders(text: str):
    ret = {}
    encoders = get_encoders()
    logger.info(f"Benchmarking {len(encoders)} token encoders")
    for encoder in encoders:
        logger.info(f"Benchmarking encoder {encoder}")
        t = datetime.datetime.now()
        token_count = count_tokens(text, encoder)
        duration = datetime.datetime.now() - t
        ret[encoder] = {"encoder": encoder, "count": token_count, "duration": duration}
    return ret


def main(args: argparse.Namespace) -> None:
    if args.list:
        print(f"Available encoders: {', '.join(get_encoders()).strip()}")
        sys.exit(0)

    if args.benchmark:
        benchmark = benchmark_encoders(process_input(args.text))
        sorted_benchmark = sorted(benchmark.items(), key=lambda x: x[1]["duration"], reverse=False)
        i = 1
        for encoder in sorted_benchmark:
            print(f"Encoder: {encoder[0]} - Rank: {i}/{len(sorted_benchmark)}")
            print(f"Token Count: {encoder[1]['count']}")
            print(f"Duration: {encoder[1]['duration'].total_seconds()}\n")
            i += 1

        sys.exit(0)

    if args.model:
        try:
            args.encoder = get_encoder_by_model(args.model)
            logger.info(f"Using encoder '{args.encoder}' for model '{args.model}'")
        except EncodingNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    input_text = process_input(args.text)
    logger.debug(f"Input size: {len(input_text)} characters")

    start_time = datetime.datetime.now()
    token_count = count_tokens(input_text, encoder=args.encoder)
    duration = datetime.datetime.now() - start_time

    if token_count is None:
        logger.error("Failed to count tokens")
        sys.exit(1)

    logger.debug(f"Token count: {token_count}")
    logger.debug(f"Processing time: {duration.total_seconds():.4f} seconds")

    verbose_output = f"Encoder: {args.encoder}\nInput size: {len(input_text)} characters\nProcessing time: {duration.total_seconds():.4f} seconds\nToken count: {token_count}"
    standard_output = f"Token count: {token_count}"

    if args.output:
        with open(args.output, 'w') as f:
            if args.verbose:
                f.write(verbose_output)
            elif args.quiet:
                f.write(str(token_count))
            else:
                f.write(standard_output)

        logger.info(f"Results written to {args.output}")

    if args.quiet and not args.verbose:
        print(token_count)

    if not args.quiet and not args.verbose:
        print(standard_output)
    elif args.verbose:
        print(f"Encoder: {args.encoder}")
        print(f"Input size: {len(input_text)} characters")
        print(f"Processing time: {duration.total_seconds():.4f} seconds")
        print(f"Token count: {token_count}")
    elif args.quiet:
        # Communicate the token count through the system exit code ;)
        sys.exit(token_count)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Count tokens in text using various encoders.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-t", "--text", nargs='+', help="Text file(s) to count tokens from.")
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument("-l", "--list", action='store_true', help="List available encoders.", default=False)
    mode_group.add_argument("-b", "--benchmark", action='store_true', help="Benchmark available encoders.",
                            default=False)
    parser.add_argument("-m", "--model", help="Specify an encoder based on the OpenAI model name")
    parser.add_argument("-e", "--encoder", default='cl100k_base', help="Token encoder to use")
    parser.add_argument("-o", "--output", help="Output file to write results")
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument("-v", "--verbose", action='store_true', help="Increase output verbosity")
    verbosity_group.add_argument("-q", "--quiet", action='store_true', help="Decrease output verbosity")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)

    main(args)
