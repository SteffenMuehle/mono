#source ~/.zshrc_moia
source ~/.zshrc_setd

#############
# OH-MY-ZSH #
#############

export ZSH="$HOME/.oh-my-zsh"
source $ZSH/oh-my-zsh.sh

# plugins
plugins=(history-substring-search)

# Example format: plugins=(rails git textmate ruby lighthouse)
# Uncomment the following line to use hyphen-insensitive completion.
# Case-sensitive completion must be off. _ and - will be interchangeable.
HYPHEN_INSENSITIVE="true"

# Uncomment the following line to enable command auto-correction.
ENABLE_CORRECTION="true"

# set format of command execution timestamp shown in the history command output.
HIST_STAMPS="yyyy-mm-dd"


#########
# SHELL #
#########

EDITOR='nano'

# history search
bindkey '^[[A' history-substring-search-up
bindkey '^[[B' history-substring-search-down

# History settings
HISTSIZE=100000                  # Maximum number of history entries in memory
SAVEHIST=100000                  # Maximum number of history entries saved to file
HISTFILE=~/.zsh_history          # File to save history


###########
# ALIASES #
###########
alias ls='eza --sort type --icons --git'
alias l='eza --all --sort type --icons --git'
alias cc='cd ..'
alias ccc='cd ..; cd ..'
alias cccc='cd ..; cd ..; cd ..'
alias path='echo $PATH | tr ":" "\n"'
alias ez='code -n ~/.zshrc'
alias sz='source ~/.zshrc'
alias p2n='poetry run jupytext --to notebook'
alias n2p='poetry run jupytext --to py:percent'
alias apt='sudo apt'


########
# PATH #
########

# add ~.local/bin to beginning of PATH
export PATH="$HOME/.local/bin:$PATH"

# activate mise
eval "$(/usr/bin/mise activate zsh)"


#############
# FUNCTIONS #
#############
# Function to get the current Git branch
gitb() {
  git rev-parse --abbrev-ref HEAD 2> /dev/null
}

# write function 'cpd' (CoPy Directory) used as for example 'cpm 3' that takes the current directory (say /home/user/Downloads)
# and creates the alias/variable 'd3' that takes holds that directory.
# so for example one could then "cd $d3" to go to /home/user/Downloads or "ls $d3" to list the contents of /home/user/Downloads
setdir() {
  local dir=$(pwd)
  local num=$1
  local var="dir$num"
  eval "$var=$dir"
  echo "Created variable $var with value $dir"
  alias "cd$num"="cd $dir"
  echo "Created alias cd$num for changing to $dir"
  sed -i "/alias cd$num/d" ~/.zshrc_setd
  sed -i "/$var=/d" ~/.zshrc_setd
  echo "alias cd$num=\"cd $dir\"" >> ~/.zshrc_setd
  echo "$var=\"$dir\"" >> ~/.zshrc_setd
  echo "Saved alias and variable in ~/.zshrc_setd"
}

##########
# PROMPT #
##########

COLOR_DEF=$'%f'
COLOR_BANNER=$'%F{243}'
COLOR_CWD=$'%F{208}'
COLOR_GIT=$'%F{42}'
COLOR_FIRST_COL=$'%F{010}'
setopt PROMPT_SUBST
NEWLINE=$'\n'

# Update PROMPT to use the dynamic banner and aligned hashes

# Function to calculate the maximum length dynamically
calculate_max_length() {
  local dir_length=${#${(%):-%~}}
  # get branch name by calling gitb function
  local branch_name=$(gitb)
  local gitb_length=${#branch_name}
  echo $(( dir_length > gitb_length ? dir_length : gitb_length ))
}

# Function to generate the banner with hashes dynamically
generate_banner() {
  local max_length=$(calculate_max_length)
  local banner_length=$(( max_length + 6 ))  # +4 to account for spaces and hashes on both sides
  printf '%*s' "$banner_length" '' | tr ' ' '='
}

# Function to pad content with spaces and add a hash at the end dynamically
pad_content() {
  local content="$1"
  local max_length=$(calculate_max_length)
  local content_length=${#${(%)content}}
  local padding_length=$(( max_length - content_length ))
  printf '%s%*s' "$content" "$padding_length" ''
}

generate_prompt() {
  local ls_output=$(ls -al | awk '
    BEGIN { files=0; dirs=0 }
    /^-/ { files++ }
    /^d/ { dirs++ }
    END { print files " files, " dirs " folders" }
  ')
  local prompt="${NEWLINE}${COLOR_BANNER}$(generate_banner)"
  prompt+=" ${COLOR_BANNER}$(date "+%Y-%m-%d %H:%M:%S")
${COLOR_CWD}cwd: $(pad_content "%~")  ${COLOR_BANNER}${ls_output}"
  local git_branch=$(gitb)
  if [[ -n "$git_branch" ]]; then
    local git_status=$(git status --porcelain 2>/dev/null | awk '
      BEGIN { modified=0; deleted=0; staged=0; untracked=0 }
      /^\?\?/ { untracked++ }
      /^ M/ { modified++ }
      /^ D/ { deleted++ }
      /^A/ { staged++ }
      /^M/ { staged++ }
      END {
        status = ""
        if (modified > 0) status = status modified " modified, "
        if (deleted > 0) status = status deleted " deleted, "
        if (staged > 0) status = status staged " staged, "
        if (untracked > 0) status = status untracked " untracked, "
        if (length(status) > 0) status = substr(status, 1, length(status) - 2)
        print status
      }
    ')
    prompt+="${NEWLINE}${COLOR_GIT}git: $(pad_content "$(gitb)")  ${COLOR_BANNER}${git_status}"
  fi
  prompt+="${NEWLINE}${COLOR_BANNER}$(generate_banner)
${COLOR_DEF}> "
  echo "$prompt"
}

export PROMPT='$(generate_prompt)'