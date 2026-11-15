"""Command line: python -m earlywarning explore | train | experiments"""

import argparse
import importlib

COMMANDS = {
    "explore": "check the data and save EDA charts (seconds)",
    "train": "tune, choose by cross-validation, train, test once, save",
    "experiments": "repeated CV, ablation, learning curve, calibration, fairness, errors (after train)",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m earlywarning",
        description="Early warning after the first term (UCI Portuguese student data).")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in COMMANDS.items():
        commands.add_parser(name, help=help_text)
    command = parser.parse_args().command

    # Each command lives in the module of the same name. It is imported only
    # now, so "explore" does not wait for TensorFlow to load.
    importlib.import_module(f"earlywarning.{command}").run()


main()
