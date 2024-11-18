from pathlib import Path

import click
import moonshine

from moonshine_cli.moonshine_cli import __version__
from moonshine_cli.moonshine_cli.loader import AudioChunkLoader

DEFAULT_BOOK_PATH = Path(
    "/home/arelius/workspace/coqui-dataset-pipeline/data/Uncrowned.mp3"
)


@click.group(help="moonshine-cli CLI Application")
@click.version_option(version=__version__)
def moonshine_cli() -> None:
    pass


@moonshine_cli.command(name="transcribe", help="Transcribes provided audio file")
@click.option(
    "--audio-file-path",
    default=DEFAULT_BOOK_PATH,
    show_default=False,
)
def transcribe(audio_file_path: Path) -> None:
    loader = AudioChunkLoader(
        audio_path=audio_file_path, chunk_duration=10.0, sr=22050, overlap=0.0
    )
    for _, chunk in enumerate(loader):
        print(chunk)

        output = moonshine.transcribe(chunk, "moonshine/tiny")

        print(output)


if __name__ == "__main__":
    moonshine_cli(prog_name="moonshine-cli")
