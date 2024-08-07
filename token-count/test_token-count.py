import io
import os
import tempfile
import unittest
from unittest.mock import patch, mock_open, MagicMock

# Import the functions from your main script
from token_count import get_encoders, get_encoder_by_model, count_tokens, process_input, main, EncodingNotFoundError


class TestTokenCounter(unittest.TestCase):

    def setUp(self):
        # self.lorem_ipsum = """
        # Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        # Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
        # Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
        # Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
        # """
        TESTFILES = ["test_lorem.txt", "test_lorem-big.txt"]
        with open(TESTFILES[1]) as f:
            self.lorem_ipsum = f.read()

    def test_get_encoders(self):
        encoders = get_encoders()
        self.assertIsInstance(encoders, list)
        self.assertIn('cl100k_base', encoders)

    def test_get_encoder_by_model(self):
        encoder = get_encoder_by_model('gpt-3.5-turbo')
        self.assertEqual(encoder, 'cl100k_base')

    def test_get_encoder_by_model_negative(self):
        with self.assertRaises(EncodingNotFoundError):
            get_encoder_by_model('nonexistent-model')

    def test_count_tokens(self):
        text = "Hello, world!"
        token_count = count_tokens(text)
        self.assertIsInstance(token_count, int)
        self.assertGreater(token_count, 0)

    def test_count_tokens_negative(self):
        with patch('tiktoken.get_encoding', side_effect=KeyError):
            result = count_tokens("Test", encoder="nonexistent_encoder")
        self.assertIsNone(result)

    def test_process_input(self):
        test_input = "This is a test.\nSecond line."
        with patch('fileinput.input', return_value=io.StringIO(test_input)):
            result = process_input()
        self.assertEqual(result, test_input)

    @patch('argparse.ArgumentParser.parse_args')
    @patch('builtins.print')
    def test_main_with_text_input(self, mock_print, mock_args):
        mock_args.return_value = MagicMock(
            text=['test_lorem.txt'], list=False, model=None, encoder='cl100k_base',
            output=None, verbose=False, quiet=False
        )

        with patch('builtins.open', mock_open(read_data=self.lorem_ipsum)):
            main(mock_args.return_value)

        mock_print.assert_called_once()

    @patch('argparse.ArgumentParser.parse_args')
    @patch('builtins.print')
    def test_main_list_encoders(self, mock_print, mock_args):
        mock_args.return_value = MagicMock(
            text=None, list=True, model=None, encoder='cl100k_base',
            output=None, verbose=False, quiet=False
        )

        main(mock_args.return_value)

        mock_print.assert_called()

    @patch('argparse.ArgumentParser.parse_args')
    @patch('builtins.print')
    def test_main_with_model(self, mock_print, mock_args):
        mock_args.return_value = MagicMock(
            text=['test_lorem.txt'], list=False, model='gpt-3.5-turbo', encoder=None,
            output=None, verbose=False, quiet=False
        )

        with patch('builtins.open', mock_open(read_data=self.lorem_ipsum)):
            main(mock_args.return_value)

        mock_print.assert_called_once()

    @patch('argparse.ArgumentParser.parse_args')
    @patch('builtins.print')
    def test_main_with_custom_encoder(self, mock_print, mock_args):
        mock_args.return_value = MagicMock(
            text=['test_lorem.txt'], list=False, model=None, encoder='p50k_base',
            output=None, verbose=False, quiet=False
        )

        with patch('builtins.open', mock_open(read_data=self.lorem_ipsum)):
            main(mock_args.return_value)

        mock_print.assert_called_once()

    @patch('argparse.ArgumentParser.parse_args')
    @patch('builtins.print')
    def test_main_with_output_file(self, mock_print, mock_args):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_filename = temp_file.name
            mock_args.return_value = MagicMock(
                text=['test_lorem.txt'], list=False, model=None, encoder='cl100k_base',
                output=temp_filename, verbose=False, quiet=False
            )

            with patch('builtins.open', mock_open(read_data=self.lorem_ipsum)):
                main(mock_args.return_value)

            with open(temp_filename, 'r') as f:
                content = f.read()
                self.assertIn("Token count:", content)

        os.unlink(temp_filename)

    @patch('argparse.ArgumentParser.parse_args')
    @patch('builtins.print')
    def test_main_verbose_mode(self, mock_print, mock_args):
        mock_args.return_value = MagicMock(
            text=['test_lorem.txt'], list=False, model=None, encoder='cl100k_base',
            output=None, verbose=True, quiet=False
        )

        with patch('builtins.open', mock_open(read_data=self.lorem_ipsum)):
            with self.assertLogs(level='DEBUG') as log:
                main(mock_args.return_value)

        self.assertTrue(any('DEBUG' in record.levelname for record in log.records))

    @patch('argparse.ArgumentParser.parse_args')
    @patch('builtins.print')
    def test_main_quiet_mode(self, mock_print, mock_args):
        mock_args.return_value = MagicMock(
            text=['test_lorem.txt'], list=False, model=None, encoder='cl100k_base',
            output=None, verbose=False, quiet=True
        )

        with patch('builtins.open', mock_open(read_data=self.lorem_ipsum)):
            with self.assertLogs(level='ERROR') as log:
                main(mock_args.return_value)

        self.assertEqual(len(log.records), 0)

    def test_main_with_nonexistent_file(self):
        with patch('sys.argv', ['token_counter.py', '-t', 'nonexistent.txt']):
            with self.assertRaises(SystemExit):
                with patch('sys.stderr', new=io.StringIO()) as fake_stderr:
                    main(None)

            error_output = fake_stderr.getvalue()
            self.assertIn("No such file or directory", error_output)

    def test_main_with_invalid_encoder(self):
        with patch('sys.argv', ['token_counter.py', '-t', 'test_lorem.txt', '-e', 'invalid_encoder']):
            with self.assertRaises(SystemExit):
                with patch('sys.stderr', new=io.StringIO()) as fake_stderr:
                    main(None)

            error_output = fake_stderr.getvalue()
            self.assertIn("Error creating encoder", error_output)


if __name__ == '__main__':
    unittest.main()
