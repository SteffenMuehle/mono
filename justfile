
@default: fmt lint test

folders := "playground llm-chat fitness finances"

install package='':
    #!/usr/bin/env bash
    mise install
    if [ -z '{{package}}' ]; then
        echo "Installing all packages..."
        uv sync --all-packages --locked     
    else
        echo "Installing package '{{package}}'..."
        uv sync --package '{{package}}' --locked
    fi

fmt:
    #!/usr/bin/env bash
    for folder in {{folders}}; do
        echo "Formatting $folder"
        just --justfile $folder/justfile fmt
    done

lint:
    #!/usr/bin/env bash
    for folder in {{folders}}; do
        echo "Linting $folder"
        just --justfile $folder/justfile lint
    done

test:
    #!/usr/bin/env bash
    for folder in {{folders}}; do
        echo "Testing $folder"
        just --justfile $folder/justfile test
    done

bumpdeps:
    uv lock --upgrade
    just install
