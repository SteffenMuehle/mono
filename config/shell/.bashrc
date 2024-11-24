#folder with symbolic links to binaries
export PATH="$HOME/programs/bin:$PATH"

#java path for JAVA_HOME
export JAVA_HOME=~/programs/bin/java_home

#java path to binaries
export PATH="$JAVA_HOME/bin:$PATH"


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/smuehle/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/smuehle/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/smuehle/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/smuehle/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

#aliases
alias ll='ls -alF'
alias home='cd ~'
alias cc='cd ..'
alias ccc='cd ..; cd ..'
alias cccc='cd ..; cd ..; cd ..'
alias cdscratch='cd /scratch.local/data/smuehle/'
alias tab='terminator --new-tab'
alias cdsim='cd /scratch01.local/smuehle/Codes/'
alias cddata='cd /data/smuehle'
alias cdproject='cd /project.dcf/poss/datasets/'
alias lsw='ls | wc'
alias nl='nano -l'

#functions
datacp() {
    mkdir "/scratch.local/data/smuehle/$1"
    scp "smuehle@uran001:/data/smuehle/$1/*" "/scratch.local/data/smuehle/$1"
}

juliaq() {
    first="$1"
    shift
    qsub -S /bin/bash -cwd -q titan.q -j yes -N $first -b y ~/programs/bin/julia "$first.jl" $@
}


# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
        # We have color support; assume it's compliant with Ecma-48
        # (ISO/IEC-6429). (Lack of such support is extremely rare, and such
        # a case would tend to support setf rather than setaf.)
        color_prompt=yes
    else
        color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt


# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi
