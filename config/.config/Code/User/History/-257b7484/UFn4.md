# Fresh "Debian Bookworm"
On a different device, prepare a USB stick with a bootable ISO of Debian Bookworm


## 1. Sudo access
su -
    add user 'debian' to sudo users
sudo apt update

## 2. regolith

1. install

2. configure


## 3. Firefox password access
- open firefox
- top-right: log in


## 4. Github access

### SSH
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

### alternative: PAT
> git config --global credential.helper store
In Firefox:
- log into github
- settings --> developer settings --> Personal access tokens --> Tokens (classic)
- generate new token, copy it
- do something in local git, it will ask user (=email) + password (enter token)


## 5. Clone config repo
```
cd
mkdir repos
cd repos
git clone https://github.com/SteffenMuehle/config
```

## 6. Copy personal data from USB-stick
- copy into `~/personal/data`
1

## 7. Install programs

### zsh
```
sudo apt install zsh
chsh -s $(which zsh)
```

### curl
sudo apt install curl

### tldr
sudo apt install tldr

### eza
https://github.com/eza-community/eza/blob/main/INSTALL.md

install nerdfont:
https://github.com/ryanoasis/nerd-fonts?tab=readme-ov-file#font-installation
download: https://www.nerdfonts.com/font-downloads
> cp ~/Downloads/font.zip ~/.local/share/fonts
> unzip .local/share/fonts/font.zip
> fc-cache -fv

add to file ~/.config/regolith3/Xresources:
> gnome.terminal.font: RobotoMono Nerd Font 14

add to vscode's settings.json:
> "terminal.integrated.fontFamily": "'RobotoMono Nerd Font', 'Courier New', monospace",

### fzf

- log out and in

### vscode
sudo apt install tldr

### thunderbird
> sudo apt install thunderbird
- account setup: get email+pw from firefox
- calendars: read in from `~/personal/data/calendar`


### mise


## 7. Set up monorepo
```
git clone https://github.com/SteffenMuehle/mono
```