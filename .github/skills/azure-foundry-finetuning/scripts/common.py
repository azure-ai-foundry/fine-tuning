import argparse
import sys


class HelpOnErrorParser(argparse.ArgumentParser):
    """ArgumentParser that prints full help when parsing fails."""

    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, f"\nerror: {message}\n")