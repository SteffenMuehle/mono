# Fresh "Debian Bookworm"
On a different device, prepare a USB stick with a bootable ISO of Debian Bookworm

## Sudo access
su -
    add user 'debian' to sudo users
sudo apt update

## Firefox password access
- open firefox
- top-right: log in

## Github access

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

## Install programs

### tldr
sudo apt install tldr

### zsh

### vscode
sudo apt install tldr

### thunderbird
> sudo apt install thunderbird
- account setup:

### regolith

1. install

2. configure

### mise


## monorepo
