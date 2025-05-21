# TODO
set up minimal .zshrc for substring search shared
https://askubuntu.com/questions/23630/how-do-you-share-history-between-terminals-in-zsh

# Requirements
we need:
- new laptop, PC or VM
- external data backup device
- Wifi and its password
- phone for
    - Firefox password
    - Github app for login 2FA
    - coupling Signal app

# 1. Install OS
On a different device, prepare a USB stick or SSD with a bootable ISO of OS

## Pop-OS
1. get display scaling right and deactivate HiDPI daemon (in display settings)
2. always display battery percentage
3. install gnome extension via browser to display workspace number
4. customize OS keyboard shortcuts


# 1. Copy data


# 2. Firefox

## 2.1 password access
- open firefox
- top-right: log in
- get password from Phone.

## 2.2 Adblocker
ublock origin, just google it + install_button

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


# 3.2 config git
> cp ~/repos/mono/config/git/.gitconfig ~

# 4. zsh, eza, gnome-tweaks, nerdfont

## install zsh
```
sudo apt install zsh
chsh -s $(which zsh)
```
log out, log in

## install oh my zsh
https://github.com/ohmyzsh/ohmyzsh/wiki
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

## use custom .zshrc
cp ~/repos/mono/config/shell/.zshrc ~

## eza
https://github.com/eza-community/eza/blob/main/INSTALL.md:

```
sudo mkdir -p /etc/apt/keyrings
wget -qO- https://raw.githubusercontent.com/eza-community/eza/main/deb.asc | sudo gpg --dearmor -o /etc/apt/keyrings/gierens.gpg
echo "deb [signed-by=/etc/apt/keyrings/gierens.gpg] http://deb.gierens.de stable main" | sudo tee /etc/apt/sources.list.d/gierens.list
sudo chmod 644 /etc/apt/keyrings/gierens.gpg /etc/apt/sources.list.d/gierens.list
sudo apt update
install -y eza
```

## gnome-tweaks
> install gnome-tweaks

## nerd-fonts
https://github.com/ryanoasis/nerd-fonts?tab=readme-ov-file#font-installation
download: https://www.nerdfonts.com/font-downloads
```
mkdir ~/.local/share/fonts
cp ~/Downloads/RobotoMono.zip ~/.local/share/fonts
cd ~/.local/share/fonts
unzip RobotoMono.zip
fc-cache -fv
```

[needed on debian 12, not on ubuntu 24]
add to file ~/.config/regolith3/Xresources:
> gnome.terminal.font: RobotoMono Nerd Font 14


# german locale for date format
maybe it was enough to go to "Settings > Date & Time", but I did
> sudo locale-gen de_DE.UTF-8
first.
see also https://www.server-world.info/en/note?os=Ubuntu_24.04&p=locale


# Install programs
> install fzf
> install ranger
> install just
> install graphviz

## code
download .deb file from their website, then
> cd ~/Downloads
> install ./code....

add to vscode's settings.json:
> "terminal.integrated.fontFamily": "'RobotoMono Nerd Font', 'Courier New', monospace",


## thunderbird
> install thunderbird
- account setup: get email+pw from firefox
- date & time in German should already work


## mise
https://mise.jdx.dev/installing-mise.html:

I had to run this twice:
```
sudo apt update -y && sudo apt install -y gpg sudo wget curl
sudo install -dm 755 /etc/apt/keyrings
wget -qO - https://mise.jdx.dev/gpg-key.pub | gpg --dearmor | sudo tee /etc/apt/keyrings/mise-archive-keyring.gpg 1> /dev/null
echo "deb [signed-by=/etc/apt/keyrings/mise-archive-keyring.gpg arch=amd64] https://mise.jdx.dev/deb stable main" | sudo tee /etc/apt/sources.list.d/mise.list
sudo apt update
sudo apt install -y mise
```


## Signal
```
# NOTE: These instructions only work for 64-bit Debian-based
# Linux distributions such as Ubuntu, Mint etc.

# 1. Install our official public software signing key:
wget -O- https://updates.signal.org/desktop/apt/keys.asc | gpg --dearmor > signal-desktop-keyring.gpg
cat signal-desktop-keyring.gpg | sudo tee /usr/share/keyrings/signal-desktop-keyring.gpg > /dev/null

# 2. Add our repository to your list of repositories:
echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/signal-desktop-keyring.gpg] https://updates.signal.org/desktop/apt xenial main' |\
  sudo tee /etc/apt/sources.list.d/signal-xenial.list

# 3. Update your package database and install Signal:
sudo apt update && sudo apt install signal-desktop
```


## Dropbox
at website, download .deb installer
cd Downloads && install ./dro.....


## Obsidian
at website, download .deb installer
cd Downloads && install ./dro.....

symlink Dropbox 'shared_resources' into ~/data/obsidian