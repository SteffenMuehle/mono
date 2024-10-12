## Installation guide

To set this repo up on your local machine, follow the following guide.

### Install mise
As a tool version manager, we use `mise`.
Install it via

```bash
brew install mise
```

and then add its activation-step into your shell:
```bash
echo 'eval "$(mise activate zsh)"' >> ~/.zshrc
```

Now continue in a new shell session, or source your startup script:
```bash
source ~/.zshrc
```

### Clone repo
Move to the folder where you want to install this repo and clone it, i.e.:
```bash
cd ~/repos
git clone https://github.com/SteffenMuehle/mono
```

### Install tools via mise

Mise is used to handle the used `Python` version, as well as the used `Poetry` version.
Poetry is the Python package manager we use.
The used versions of tools handled by mise are specified in the `.tool-versions` file.

Make the specified versions available to Mise on your machine via moving to the repo and installing them like so:
```bash
mise install
```

This needs to be re-run whenever the version inside the `.tool-versions` file have been updated.
To confirm that the tool versions have been added to your PATH variable you can run
```bash
echo $PATH | tr ":" "\n"
```

The first lines should look like `/Users/steffen.muehle/.local/share/mise/installs/python/3.12.4/bin`.

### Configure poetry

- Retrieve your artifactory API credentials from https://moiadev.jfrog.io/ui/user_profile and store them in your env variables `ARTIFACTORY_USER` and `ARTIFACTORY_APIKEY`.

```bash
poetry config virtualenvs.in-project true
```

### Install Python environment via poetry

For each python package (subfolder of repository), we need to install the python dependencies onto your local machine
in a virtual environment:

```bash
for folder in fitness; do
    cd $folder
    poetry install
    cd ..
done
```

### Run a notebook

Jupyter notebooks in this repository are not checked into git as .ipynb files,
but as .py files ("percent scripts"), see https://jupytext.readthedocs.io/en/latest/formats-scripts.html.
The recommended workflow is:
1. Locally convert the .py files into .ipynb (notebooks) via `cd optimizer` and `just notebook`
2. Do your work in the .ipynb files and save them to disk
3. Sync your changes to the .py file via `just notebook-sync`
4. Commit your .py file
5. The .ipynb is git-ignored, you can delete it if you want

#### Open notebook via IDE (recommended)

1. Vscode\
Install the following extensions:
- `Jupyter`

and open the .ipynb files in vscode.

When running a notebook for the first time, you need to select the correct Python interpreter for it. That should be the interpreter in the same package's `.venv/bin/python`.

Each interpreter needs to be manually registered with vscode once. You can do this by
- in the IDE's file tree, locate e.g. optimizer/.venv/bin/python
- right click on it and copy its full path
- open vscode's command palette (ctrl+shift+p, or View-->"Command Palette")
- type/click on "Python: Select Interpreter", then "Enter interpreter path..."
- paste the full path you copied and hit Enter.
Now this Python interpreter should be choosable when clicking the opened notebooks top-right "Select Interpreter" button
