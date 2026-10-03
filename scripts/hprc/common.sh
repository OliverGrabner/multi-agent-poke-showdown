# Shared settings for the HPRC job scripts. Sourced, not run.
# Installed tools, environments and model weights live in the checkout (research scratch).
# Download caches go to the node's temporary disk: scratch allows 250,000 files and $HOME only
# 10,000, and package caches alone run to tens of thousands of files.
export PATH="$PWD/.tools:$PWD/.tools/node/bin:$PATH"
node_tmp="${TMPDIR:-/tmp}/pokerl-$USER"
export UV_CACHE_DIR="$node_tmp/uv"
export npm_config_cache="$node_tmp/npm"
export UV_PYTHON_INSTALL_DIR="$PWD/cache/python"
export UV_MANAGED_PYTHON=true
export XDG_CACHE_HOME="$PWD/cache"
export HF_HOME="$PWD/cache/huggingface"
export FLASHINFER_WORKSPACE_BASE="$PWD/cache"
# Battles talk to the model server on this node; keep that traffic away from the web proxy.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
mkdir -p runs
