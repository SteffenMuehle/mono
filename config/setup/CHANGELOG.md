1. Update apt
> sudo apt update


2. Install curl
> sudo apt install curl


3. Install brew + dependencies

3.1 Download via curl
ref: https://brew.sh
> /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

3.2 follow the next steps:
```
Next steps:
- Run these two commands in your terminal to add Homebrew to your PATH:
    (echo; echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"') >> /home/ksteffen/.bashrc
    eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
- Install Homebrew's dependencies if you have sudo access:
    sudo apt-get install build-essential
  For more information, see:
    https://docs.brew.sh/Homebrew-on-Linux
- We recommend that you install GCC:
    brew install gcc
- Run brew help to get started
- Further documentation:
    https://docs.brew.sh
```


4. Install zsh
ref: https://github.com/ohmyzsh/ohmyzsh/wiki/Installing-ZSH

4.1 Install
> brew install zsh

4.2 Make default shell
- add the output of
> which zsh

to /usr/shells via
> sudo nano /usr/shells

then change default shell via
> chsh -s $(which zsh)


5. Restart + start shell
> go through zsh setup menu


6. Clone configs from GitHub
> cd; mkdir code
> git clone https://github.com/SteffenMuehle/config


7. Configure .zsh
> cd; cp code/config/.zshrc .zshrc
> source .zshrc


8. Install tldr
> brew install tlrc


9. Configure git
9.1 global git config
> cd; cp code/config/.gitconfig ~
9.2 git lfs
> brew install git-lfs
> git lfs install
9.3 Personal Access Token
Generate PAT on GitHub: https://github.com/settings/tokens
> git config --global credential.helper store
> cd ~/code/config
[MAKE A COMMIT]
> git push
> [user = steffen_muehle@gmx.de]
> [pw = PAT]

o10. Install vscode
10.1 Download .deb file from website:
https://code.visualstudio.com/docs/setup/linux

10.2 Install via apt
> cd; cd Downloads; sudo apt install ./code_1.89.0-1714530869_amd64.deb

10.3 Extensions
--skip--


11. Install direnv
> sudo apt install direnv


11. Install asdf
11.1 clone git repo into home
> git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.14.0

11.2 configure use of asdf in .zshrc
add:
"""
. "$HOME/.asdf/asdf.sh"
# append completions to fpath
fpath=(${ASDF_DIR}/completions $fpath)
# initialise completions with ZSH's compinit
autoload -Uz compinit && compinit
"""


12 install build dependencies for asdf to use

12.1 python
check required python build dependencies for your OS:
https://github.com/pyenv/pyenv/wiki#suggested-build-environment
> sudo apt update; sudo apt install build-essential libssl-dev zlib1g-dev \
> libbz2-dev libreadline-dev libsqlite3-dev curl \
> libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev


12. Clone/setup dev_repo
> cd ~/code; mkdir dev-repo; cd dev-repo
> git init

12.1 Install Python3.11 via asdf
> echo "python 3.11.0" > .tool-versions
> cat .tool-versions | while IFS=' ' read t v ; do asdf plugin add $t ; asdf install $t $v ; done


13. 
toolsversions:
- poetry
- python 3.11


# TODO
- [ ] tldr
- [ ] git
- zshrc
- gitconfig
- code
- code settings
- asdf
- poetry
- pythonxx
- rustyy
- pre-commit
- linters
    - ruff
    - mypy
- pydantic
- just
- pytest
- GH repo w/ tests
- Jenkings, Travis
