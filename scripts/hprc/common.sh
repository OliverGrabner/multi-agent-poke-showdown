# Shared settings for the HPRC job scripts. Sourced, not run.
# Tools and caches live inside the checkout (research scratch), never in $HOME.
export PATH="$PWD/.tools:$PWD/.tools/node/bin:$PATH"
export UV_CACHE_DIR="$PWD/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/cache/python"
export UV_MANAGED_PYTHON=true
export XDG_CACHE_HOME="$PWD/cache"
export HF_HOME="$PWD/cache/huggingface"
export FLASHINFER_WORKSPACE_BASE="$PWD/cache"
# Battles talk to the model server on this node; keep that traffic away from the web proxy.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
mkdir -p runs
