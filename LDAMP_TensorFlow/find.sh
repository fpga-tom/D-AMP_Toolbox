find . -name '*.c' -exec /bin/bash -c 'clang -Xclang -dump-tokens {} 2>&1 | cut -d " "  -f 1 | grep -v "\." | grep -v "/" | grep -v "#" | grep -v "[0-9]" ' \; > tokens.txt
