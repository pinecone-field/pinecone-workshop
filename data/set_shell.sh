#!/bin/bash
. /root/.bash_aliases
python -m venv myenv
. myenv/bin/activate
poetry shell