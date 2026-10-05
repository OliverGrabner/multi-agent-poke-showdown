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
# Compiled GPU kernels are kept between jobs so later servers start faster.
export FLASHINFER_WORKSPACE_BASE="$PWD/cache"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor"
# The vLLM environment is stored as one archive (scratch is slow at opening many small files, and
# limits their number); each job unpacks it to the node's local disk.
export VLLM_ARCHIVE="$PWD/cache/venv-vllm.tar"
# Battles talk to the model server on this node; keep that traffic away from the web proxy.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
mkdir -p runs

# The Hugging Face repo, the URL variable the battle code reads, the GPUs to use and the vLLM
# parsers for each model this project serves (sets repo, url_var, gpus and parsers).
model_info() {
  case "$1" in
    qwen3.8-27b)
      repo=Qwen/Qwen3.8-27B url_var=QWEN_BASE_URL gpus=2
      parsers=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder) ;;
    qwen3.5-2b)
      repo=Qwen/Qwen3.5-2B url_var=QWEN_SMALL_BASE_URL gpus=1
      parsers=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder) ;;
    gemma-4-31b)
      repo=google/gemma-4-31B-it url_var=GEMMA_BASE_URL gpus=2
      parsers=(--reasoning-parser gemma4 --enable-auto-tool-choice --tool-call-parser gemma4) ;;
    *)
      printf 'Unknown model %s\n' "$1" >&2
      return 2 ;;
  esac
}
