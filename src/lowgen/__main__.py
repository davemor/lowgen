import click
from dotenv import load_dotenv

from .cli import synth

load_dotenv()


@click.group(
    help="CLI tool for generating synthetic low-dimensional datasets and evaluating them."
)
def cli():
    pass


cli.add_command(synth)


def main():
    cli(prog_name="hproj")


if __name__ == "__main__":
    main()
