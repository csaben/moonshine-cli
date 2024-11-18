# moonshine-cli

use moonshine speech to text model to create transcripts via cli

## quickstart
```
uv venv moonshine-cli --python 3.11
source moonshine-cli/bin/activate
uv pip install useful-moonshine@git+https://github.com/usefulsensors/moonshine.git
export KERAS_BACKEND=torch
moonshine-cli --help
```