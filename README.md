we need:
- new laptop, PC or VM
- external data backup device
- Wifi and its password
- phone for
    - Firefox password
    - Github app for login 2FA

# 1. Install OS
On a different device, prepare a USB stick or SSD with a bootable ISO of OS

## Pop-OS
1. get display scaling right and deactivate HiDPI daemon (in display settings)
2. always display battery percentage
3. install gnome extension via browser to display workspace number
4. customize OS keyboard shortcuts


# 1. Copy data


# 2. Firefox password access
- open firefox
- top-right: log in
- get password from Phone.


# 3. Github access

## SSH
```
cd ~/.ssh
ssh-keygen -t ed25519 -C "steffen_muehle@gmx.de"
cat id_ed25519.pub

    [[COPY (including email)]]
```

In Firefox:
- log into github
- settings --> SSH and GPG keys
- new SSH key

```
cd
mkdir repos
cd repos
git clone git@github.com:SteffenMuehle/mono.git
[ MAKE COMMIT, THEN FIRST TIME: ]
git push -u origin main
```


## alternative: PAT
> git config --global credential.helper store
In Firefox:
- log into github
- settings --> developer settings --> Personal access tokens --> Tokens (classic)
- generate new token, copy it
- do something in local git, it will ask user (=email) + password (enter token)

```
cd
mkdir repos
cd repos
git clone https://github.com/SteffenMuehle/config
```


## 7. Install programs

### zsh
```
sudo apt install zsh
chsh -s $(which zsh)
```
log out, log in

oh my zsh

### tldr
sudo apt install tldr

### eza
https://github.com/eza-community/eza/blob/main/INSTALL.md:

sudo mkdir -p /etc/apt/keyrings
wget -qO- https://raw.githubusercontent.com/eza-community/eza/main/deb.asc | sudo gpg --dearmor -o /etc/apt/keyrings/gierens.gpg
echo "deb [signed-by=/etc/apt/keyrings/gierens.gpg] http://deb.gierens.de stable main" | sudo tee /etc/apt/sources.list.d/gierens.list
sudo chmod 644 /etc/apt/keyrings/gierens.gpg /etc/apt/sources.list.d/gierens.list
sudo apt update
sudo apt install -y eza



install nerdfont:
https://github.com/ryanoasis/nerd-fonts?tab=readme-ov-file#font-installation
download: https://www.nerdfonts.com/font-downloads
> mkdir ~/.local/share/fonts
> cp ~/Downloads/RobotoMono.zip ~/.local/share/fonts
> cd ~/.local/share/fonts
> unzip RobotoMono.zip
> fc-cache -fv

[needed on debian 12, not on ubuntu 24]
add to file ~/.config/regolith3/Xresources:
> gnome.terminal.font: RobotoMono Nerd Font 14

add to vscode's settings.json:
> "terminal.integrated.fontFamily": "'RobotoMono Nerd Font', 'Courier New', monospace",

### fzf
sudo apt install fzf

### ranger
sudo apt install ranger

### vscode
download .deb file from their website, then
cd ~/Downloads
sudo apt install ./code....


### thunderbird
> sudo apt install thunderbird
- account setup: get email+pw from firefox
- calendars: read in from `~/personal/data/calendar`

datetime format yyyy-MM-dd:
https://support.mozilla.org/en-US/kb/customize-date-time-formats-thunderbird

get dark mode extension


### mise
https://mise.jdx.dev/installing-mise.html:

apt update -y && apt install -y gpg sudo wget curl
sudo install -dm 755 /etc/apt/keyrings
wget -qO - https://mise.jdx.dev/gpg-key.pub | gpg --dearmor | sudo tee /etc/apt/keyrings/mise-archive-keyring.gpg 1> /dev/null
echo "deb [signed-by=/etc/apt/keyrings/mise-archive-keyring.gpg arch=amd64] https://mise.jdx.dev/deb stable main" | sudo tee /etc/apt/sources.list.d/mise.list
sudo apt update
sudo apt install -y mise


https://github.com/asdf-community/asdf-python/issues/119:

sudo apt install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

## hibernate
https://ubuntuhandbook.org/index.php/2021/08/enable-hibernate-ubuntu-21-10/
https://ubuntuhandbook.org/index.php/2020/05/lid-close-behavior-ubuntu-20-04/

## german locale for date format ere nad there
https://www.server-world.info/en/note?os=Ubuntu_24.04&p=locale

## chatgpt desktop
https://snapcraft.io/install/chatgpt-desktop/ubuntu

## regolith config

## 7. Set up monorepo
```
git clone https://github.com/SteffenMuehle/mono
```


# todo
- regolith autolaunch apps upon launch
- .. and into dedicated workspace
- vscode keybindings
- calendar sync with debian
- fitness: add 'runs' to csv
- fitness: add CLI tool for adding entries to csvtldr

fzf

eza

ranger

oh-my-posh
https://ohmyposh.dev/docs/installation/prompt

brew install zsh-history-substring-search  (https://github.com/zsh-users/zsh-history-substring-search)
