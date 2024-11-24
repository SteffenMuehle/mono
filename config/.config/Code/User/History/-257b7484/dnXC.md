# Fresh "Debian Bookworm"
On a different device, prepare a USB stick with a bootable ISO of Debian Bookworm


## 1. Sudo access
su -
    add user 'debian' to sudo users
sudo apt update


## 2. Firefox password access
- open firefox
- top-right: log in


## 3. Github access

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


## 4. Clone config repo
```
cd
mkdir repos
cd repos
git clone https://github.com/SteffenMuehle/config
```

## 5. Copy personal data from USB-stick
- copy into `~/personal/data`
1

## 6. Install programs

### tldr
sudo apt install tldr

### zsh
sudo apt install zsh
chsh -s $(which zsh)

### vscode
sudo apt install tldr

### thunderbird
> sudo apt install thunderbird
- account setup: get email+pw from firefox
- calendars: read in from `~/personal/data/calendar`

### regolith

1. install

2. configure

### mise


## 7. Set up monorepo
```
git clone https://github.com/SteffenMuehle/mono
```