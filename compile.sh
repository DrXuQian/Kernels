#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMMAND="build"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"
GPU_ARCH="${GPU_ARCH:-sm_90a}"
LINEAR_ARCH="${LINEAR_ARCH:-$GPU_ARCH}"
MARLIN_ARCH="${MARLIN_ARCH:-sm_80}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR_NAME="${BUILD_DIR_NAME:-build_cmake_release}"
CUDA_ROOT="${CUDA_ROOT:-}"
PPU_ROOT="${PPU_ROOT:-}"
CUTLASS_DIR="${CUTLASS_DIR:-$ROOT_DIR/third_party/cutlass}"
DRY_RUN=0
CHECK_ENV=1
VERBOSE=0
NVCC=""

usage() {
  cat <<'EOF'
Usage:
  ./compile.sh [command] [targets...] [options]

Commands:
  build       Build targets. This is the default command.
  configure   Run CMake configure for CMake-based targets. Makefile targets are skipped.
  clean       Clean targets.
  rebuild     Clean then build targets.
  list        List supported targets.
  env         Print and validate the build environment.

Common examples:
  ./compile.sh list
  ./compile.sh env
  ./compile.sh build
  ./compile.sh build general linear_attention
  ./compile.sh build w4a16-fpa w4a16-machete
  ./compile.sh configure w4a16-fpa
  ./compile.sh rebuild moe-vllm
  ./compile.sh clean all

Targets:
  default              Root Makefile targets: general, linear_attention, vLLM MoE CUDA pieces
  all                  Every target listed below

  general              general/
  linear_attention     linear_attention/
  flashinfer-gdn       linear_attention/src/flashinfer_gdn/

  moe-vllm             moe-w4a16 vLLM marlin + auxiliary
  moe-vllm-marlin      moe_w4a16/vllm/marlin/
  moe-vllm-auxiliary   moe_w4a16/vllm/auxiliary/
  moe-trtllm           moe_w4a16/trtllm/moe_w4a16_standalone/
  moe-trtllm-auxiliary moe_w4a16/trtllm/auxiliary/
  moe                  moe-vllm + moe-trtllm

  w4a16-marlin         w4a16_gemm/marlin_standalone/
  w4a16-fpa            w4a16_gemm/fpA_intB_standalone/
  w4a16-machete        w4a16_gemm/machete_standalone/
  w4a16-cutlass55      w4a16_gemm/cutlass55_standalone/
  w4a16-cublas         w4a16_gemm/cublas_bf16_bench.cu
  w4a16                all w4a16 targets above

Options:
  -j, --jobs N              Parallel build jobs. Default: nproc.
  --arch ARCH              Main CUDA arch. Default: sm_90a.
  --linear-arch ARCH       Arch for linear/general Makefile targets. Default: --arch.
  --marlin-arch ARCH       Arch for legacy standalone Marlin. Default: sm_80.
  --build-type TYPE        CMake build type. Default: Release.
  --build-dir-name NAME    CMake build dir basename. Default: build_cmake_release.
  --cuda-root DIR          CUDA-compatible SDK root. Also accepted from CUDA_ROOT.
  --ppu-root DIR           Optional companion SDK root. Also accepted from PPU_ROOT.
  --cutlass-dir DIR        CUTLASS checkout. Default: third_party/cutlass.
  --dry-run                Print commands without running them.
  --no-env-check           Skip environment validation.
  -v, --verbose            Print extra environment details.
  -h, --help               Show this help.

Notes:
  The script never hardcodes machine-local SDK paths. Set CUDA_ROOT/PPU_ROOT in
  your shell or pass --cuda-root/--ppu-root when a specific SDK must be used.
EOF
}

log() {
  printf '[compile] %s\n' "$*"
}

warn() {
  printf '[compile][warn] %s\n' "$*" >&2
}

die() {
  printf '[compile][error] %s\n' "$*" >&2
  exit 1
}

quote_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    printf -v arg '%q' "$arg"
    out+=" $arg"
  done
  printf '%s' "${out# }"
}

run_cmd() {
  log "$(quote_cmd "$@")"
  if [[ "$DRY_RUN" == 0 ]]; then
    "$@"
  fi
}

run_env_cmd() {
  local -a envs=()
  while [[ $# -gt 0 && "$1" == *=* ]]; do
    envs+=("$1")
    shift
  done

  if [[ ${#envs[@]} -gt 0 ]]; then
    log "env $(quote_cmd "${envs[@]}") $(quote_cmd "$@")"
  else
    log "$(quote_cmd "$@")"
  fi

  if [[ "$DRY_RUN" == 0 ]]; then
    if [[ ${#envs[@]} -gt 0 ]]; then
      env "${envs[@]}" "$@"
    else
      "$@"
    fi
  fi
}

require_file() {
  [[ -e "$1" ]] || die "$2"
}

setup_environment() {
  if [[ -n "$CUDA_ROOT" ]]; then
    export PATH="$CUDA_ROOT/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
  fi

  if [[ -n "$PPU_ROOT" ]]; then
    export PATH="$PPU_ROOT/bin:$PATH"
    export LD_LIBRARY_PATH="$PPU_ROOT/lib:${LD_LIBRARY_PATH:-}"
  fi

  if [[ -n "$CUDA_ROOT" && -x "$CUDA_ROOT/bin/nvcc" ]]; then
    NVCC="$CUDA_ROOT/bin/nvcc"
  else
    NVCC="$(command -v nvcc || true)"
  fi
}

print_env() {
  setup_environment

  log "repo root: $ROOT_DIR"
  log "jobs: $JOBS"
  log "gpu arch: $GPU_ARCH"
  log "linear arch: $LINEAR_ARCH"
  log "marlin arch: $MARLIN_ARCH"
  log "build type: $BUILD_TYPE"
  log "build dir name: $BUILD_DIR_NAME"
  log "cutlass dir: $CUTLASS_DIR"
  log "cuda root: ${CUDA_ROOT:-<unset>}"
  log "companion sdk root: ${PPU_ROOT:-<unset>}"
  log "nvcc: ${NVCC:-<not found>}"

  if [[ -n "$NVCC" && -x "$NVCC" ]]; then
    "$NVCC" --version | sed 's/^/[compile] nvcc: /'
  fi

  if [[ "$VERBOSE" == 1 ]]; then
    command -v cmake >/dev/null 2>&1 && cmake --version | sed 's/^/[compile] cmake: /'
    command -v make >/dev/null 2>&1 && make --version | head -1 | sed 's/^/[compile] make: /'
    command -v nvcc >/dev/null 2>&1 && which -a nvcc | sed 's/^/[compile] PATH nvcc: /'
  fi
}

check_environment() {
  setup_environment

  [[ -n "$NVCC" && -x "$NVCC" ]] || die "nvcc not found. Set CUDA_ROOT or put nvcc in PATH."
  command -v make >/dev/null 2>&1 || die "make not found in PATH."
  command -v cmake >/dev/null 2>&1 || die "cmake not found in PATH."

  require_file "$CUTLASS_DIR/include/cutlass/cutlass.h" \
    "CUTLASS not found under $CUTLASS_DIR. Run: git submodule update --init third_party/cutlass"

  if [[ -n "$CUDA_ROOT" && ! -d "$CUDA_ROOT" ]]; then
    die "CUDA_ROOT does not exist: $CUDA_ROOT"
  fi

  if [[ -n "$PPU_ROOT" && ! -d "$PPU_ROOT" ]]; then
    die "PPU_ROOT does not exist: $PPU_ROOT"
  fi

  local -a nvccs=()
  while IFS= read -r line; do
    nvccs+=("$line")
  done < <(which -a nvcc 2>/dev/null | awk '!seen[$0]++')
  if [[ ${#nvccs[@]} -gt 1 ]]; then
    warn "multiple nvcc entries are visible in PATH; using $NVCC"
    if [[ "$VERBOSE" == 1 ]]; then
      printf '%s\n' "${nvccs[@]}" | sed 's/^/[compile][warn]   /' >&2
    fi
  fi
}

cccl_include_flag() {
  local cccl=""
  if [[ -n "$CUDA_ROOT" && -d "$CUDA_ROOT/targets/x86_64-linux/include/cccl" ]]; then
    cccl="$CUDA_ROOT/targets/x86_64-linux/include/cccl"
  fi
  printf '%s' "$cccl"
}

cmake_env_args() {
  if [[ -n "$CUDA_ROOT" ]]; then
    printf '%s\n' "CUDACXX=$NVCC"
  fi
}

cmake_common_args() {
  local cccl
  cccl="$(cccl_include_flag)"

  printf '%s\n' "-DCUTLASS_DIR=$CUTLASS_DIR"
  printf '%s\n' "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"

  if [[ -n "$CUDA_ROOT" ]]; then
    printf '%s\n' "-DCUDAToolkit_ROOT=$CUDA_ROOT"
  fi

  if [[ -n "$cccl" ]]; then
    printf '%s\n' "-DCMAKE_CUDA_FLAGS=-I$cccl"
    printf '%s\n' "-DCMAKE_CXX_FLAGS=-I$cccl"
  fi
}

cmake_arch_number() {
  local arch="$1"
  arch="${arch#sm_}"
  arch="${arch#compute_}"
  arch="${arch%a}"
  printf '%s' "$arch"
}

make_common_args() {
  local arch="$1"
  local cccl
  cccl="$(cccl_include_flag)"

  printf '%s\n' "NVCC=$NVCC"
  printf '%s\n' "CUDA_ROOT=$CUDA_ROOT"
  printf '%s\n' "ARCH=-arch=$arch"
  if [[ -n "$cccl" ]]; then
    printf '%s\n' "CUDA_CCCL_INC=-I$cccl"
  fi
}

build_make_dir() {
  local dir="$1"
  local arch="$2"
  shift 2
  local -a args vars
  mapfile -t vars < <(make_common_args "$arch")
  args=(-C "$ROOT_DIR/$dir" -j "$JOBS" "${vars[@]}" "$@")
  run_cmd make "${args[@]}"
}

clean_make_dir() {
  local dir="$1"
  run_cmd make -C "$ROOT_DIR/$dir" clean
}

build_linear_attention() {
  local -a vars
  mapfile -t vars < <(make_common_args "$LINEAR_ARCH")
  run_cmd make -C "$ROOT_DIR/linear_attention" -j "$JOBS" "${vars[@]}" "ARCH_SM90=-arch=$GPU_ARCH"
}

build_flashinfer_gdn() {
  local -a vars
  mapfile -t vars < <(make_common_args "$GPU_ARCH")
  run_cmd make -C "$ROOT_DIR/linear_attention/src/flashinfer_gdn" -j "$JOBS" "${vars[@]}"
}

configure_cmake_target() {
  local src="$1"
  local build="$2"
  local arch_mode="$3"

  local -a envs common configure
  mapfile -t envs < <(cmake_env_args)
  mapfile -t common < <(cmake_common_args)

  configure=(-S "$ROOT_DIR/$src" -B "$ROOT_DIR/$build" "${common[@]}")
  if [[ "$arch_mode" == "gpu_arch" ]]; then
    configure+=("-DGPU_ARCH=$GPU_ARCH")
  elif [[ "$arch_mode" == "cuda_arch_number" ]]; then
    configure+=("-DCMAKE_CUDA_ARCHITECTURES=$(cmake_arch_number "$GPU_ARCH")")
  fi

  run_env_cmd "${envs[@]}" cmake "${configure[@]}"
}

build_cmake_target() {
  local src="$1"
  local build="$2"
  local arch_mode="$3"
  shift 3

  configure_cmake_target "$src" "$build" "$arch_mode"

  local -a build_cmd
  build_cmd=(--build "$ROOT_DIR/$build" --target "$@" -j "$JOBS")
  run_cmd cmake "${build_cmd[@]}"
}

clean_cmake_dir() {
  local build="$1"
  run_cmd rm -rf "$ROOT_DIR/$build"
}

build_w4a16_cublas() {
  run_cmd "$NVCC" -O3 -std=c++17 "-arch=$GPU_ARCH" \
    "$ROOT_DIR/w4a16_gemm/cublas_bf16_bench.cu" \
    -o "$ROOT_DIR/w4a16_gemm/cublas_bf16_bench" -lcublas
}

clean_w4a16_cublas() {
  run_cmd rm -f "$ROOT_DIR/w4a16_gemm/cublas_bf16_bench"
}

configure_target() {
  case "$1" in
    default|general|linear_attention|flashinfer-gdn|moe-vllm-marlin|moe-vllm-auxiliary|moe-trtllm-auxiliary|w4a16-marlin|w4a16-cublas)
      log "$1 uses a Makefile or direct nvcc build; no CMake configure step."
      ;;
    moe-trtllm)
      configure_cmake_target \
        moe_w4a16/trtllm/moe_w4a16_standalone \
        "moe_w4a16/trtllm/moe_w4a16_standalone/$BUILD_DIR_NAME" \
        cuda_arch_number
      ;;
    w4a16-fpa)
      configure_cmake_target \
        w4a16_gemm/fpA_intB_standalone \
        "w4a16_gemm/fpA_intB_standalone/$BUILD_DIR_NAME" \
        gpu_arch
      ;;
    w4a16-machete)
      configure_cmake_target \
        w4a16_gemm/machete_standalone \
        "w4a16_gemm/machete_standalone/$BUILD_DIR_NAME" \
        gpu_arch
      ;;
    w4a16-cutlass55)
      configure_cmake_target \
        w4a16_gemm/cutlass55_standalone \
        "w4a16_gemm/cutlass55_standalone/$BUILD_DIR_NAME" \
        gpu_arch
      ;;
    *)
      die "unknown target: $1"
      ;;
  esac
}

build_target() {
  case "$1" in
    default)
      run_cmd make -C "$ROOT_DIR" -j "$JOBS" "NVCC=$NVCC" "CUDA_ROOT=$CUDA_ROOT" "ARCH=-arch=$LINEAR_ARCH" "ARCH_SM90=-arch=$GPU_ARCH"
      ;;
    general)
      build_make_dir general "$LINEAR_ARCH"
      ;;
    linear_attention)
      build_linear_attention
      ;;
    flashinfer-gdn)
      build_flashinfer_gdn
      ;;
    moe-vllm-marlin)
      build_make_dir moe_w4a16/vllm/marlin "$GPU_ARCH"
      ;;
    moe-vllm-auxiliary)
      build_make_dir moe_w4a16/vllm/auxiliary "$GPU_ARCH"
      ;;
    moe-trtllm)
      build_cmake_target \
        moe_w4a16/trtllm/moe_w4a16_standalone \
        "moe_w4a16/trtllm/moe_w4a16_standalone/$BUILD_DIR_NAME" \
        cuda_arch_number \
        test_moe_w4a16_gemm
      ;;
    moe-trtllm-auxiliary)
      build_make_dir moe_w4a16/trtllm/auxiliary "$GPU_ARCH"
      ;;
    w4a16-marlin)
      build_make_dir w4a16_gemm/marlin_standalone "$MARLIN_ARCH"
      ;;
    w4a16-fpa)
      build_cmake_target \
        w4a16_gemm/fpA_intB_standalone \
        "w4a16_gemm/fpA_intB_standalone/$BUILD_DIR_NAME" \
        gpu_arch \
        test_fpA_intB_gemm
      ;;
    w4a16-machete)
      build_cmake_target \
        w4a16_gemm/machete_standalone \
        "w4a16_gemm/machete_standalone/$BUILD_DIR_NAME" \
        gpu_arch \
        test_machete_gemm
      ;;
    w4a16-cutlass55)
      build_cmake_target \
        w4a16_gemm/cutlass55_standalone \
        "w4a16_gemm/cutlass55_standalone/$BUILD_DIR_NAME" \
        gpu_arch \
        cutlass55_fp16_gemm cutlass55_bf16_gemm
      ;;
    w4a16-cublas)
      build_w4a16_cublas
      ;;
    *)
      die "unknown target: $1"
      ;;
  esac
}

clean_target() {
  case "$1" in
    default)
      run_cmd make -C "$ROOT_DIR" clean
      ;;
    general)
      clean_make_dir general
      ;;
    linear_attention)
      clean_make_dir linear_attention
      ;;
    flashinfer-gdn)
      clean_make_dir linear_attention/src/flashinfer_gdn
      ;;
    moe-vllm-marlin)
      clean_make_dir moe_w4a16/vllm/marlin
      ;;
    moe-vllm-auxiliary)
      clean_make_dir moe_w4a16/vllm/auxiliary
      ;;
    moe-trtllm)
      clean_cmake_dir "moe_w4a16/trtllm/moe_w4a16_standalone/$BUILD_DIR_NAME"
      ;;
    moe-trtllm-auxiliary)
      clean_make_dir moe_w4a16/trtllm/auxiliary
      ;;
    w4a16-marlin)
      clean_make_dir w4a16_gemm/marlin_standalone
      ;;
    w4a16-fpa)
      clean_cmake_dir "w4a16_gemm/fpA_intB_standalone/$BUILD_DIR_NAME"
      ;;
    w4a16-machete)
      clean_cmake_dir "w4a16_gemm/machete_standalone/$BUILD_DIR_NAME"
      ;;
    w4a16-cutlass55)
      clean_cmake_dir "w4a16_gemm/cutlass55_standalone/$BUILD_DIR_NAME"
      ;;
    w4a16-cublas)
      clean_w4a16_cublas
      ;;
    *)
      die "unknown target: $1"
      ;;
  esac
}

list_targets() {
  usage | sed -n '/^Targets:/,/^Options:/p' | sed '$d'
}

expand_one_target() {
  case "$1" in
    all)
      printf '%s\n' \
        general linear_attention flashinfer-gdn \
        moe-vllm-marlin moe-vllm-auxiliary moe-trtllm moe-trtllm-auxiliary \
        w4a16-marlin w4a16-fpa w4a16-machete w4a16-cutlass55 w4a16-cublas
      ;;
    moe)
      printf '%s\n' moe-vllm-marlin moe-vllm-auxiliary moe-trtllm moe-trtllm-auxiliary
      ;;
    moe-vllm)
      printf '%s\n' moe-vllm-marlin moe-vllm-auxiliary
      ;;
    w4a16)
      printf '%s\n' w4a16-marlin w4a16-fpa w4a16-machete w4a16-cutlass55 w4a16-cublas
      ;;
    linear|linear-attention)
      printf '%s\n' linear_attention
      ;;
    flashinfer|gdn|flashinfer_gdn)
      printf '%s\n' flashinfer-gdn
      ;;
    moe-marlin|marlin-moe)
      printf '%s\n' moe-vllm-marlin
      ;;
    moe-aux|moe-auxiliary)
      printf '%s\n' moe-vllm-auxiliary
      ;;
    trtllm-moe|moe_w4a16_standalone)
      printf '%s\n' moe-trtllm
      ;;
    trtllm-aux|trtllm-auxiliary|moe-trtllm-aux)
      printf '%s\n' moe-trtllm-auxiliary
      ;;
    fpa|fpA_intB|fpA-intB|fpaintb)
      printf '%s\n' w4a16-fpa
      ;;
    machete)
      printf '%s\n' w4a16-machete
      ;;
    cutlass55)
      printf '%s\n' w4a16-cutlass55
      ;;
    cublas)
      printf '%s\n' w4a16-cublas
      ;;
    marlin)
      printf '%s\n' w4a16-marlin
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}

expand_targets() {
  local -a input=("$@")
  local -a expanded=()
  local t e seen

  for t in "${input[@]}"; do
    while IFS= read -r e; do
      seen=0
      for existing in "${expanded[@]:-}"; do
        if [[ "$existing" == "$e" ]]; then
          seen=1
          break
        fi
      done
      [[ "$seen" == 0 ]] && expanded+=("$e")
    done < <(expand_one_target "$t")
  done

  printf '%s\n' "${expanded[@]}"
}

TARGET_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    build|configure|clean|rebuild|list|env)
      COMMAND="$1"
      shift
      ;;
    --list)
      COMMAND="list"
      shift
      ;;
    -j|--jobs)
      JOBS="${2:?missing value for $1}"
      shift 2
      ;;
    --jobs=*)
      JOBS="${1#*=}"
      shift
      ;;
    --arch|--gpu-arch)
      GPU_ARCH="${2:?missing value for $1}"
      shift 2
      ;;
    --arch=*|--gpu-arch=*)
      GPU_ARCH="${1#*=}"
      shift
      ;;
    --linear-arch)
      LINEAR_ARCH="${2:?missing value for $1}"
      shift 2
      ;;
    --linear-arch=*)
      LINEAR_ARCH="${1#*=}"
      shift
      ;;
    --marlin-arch)
      MARLIN_ARCH="${2:?missing value for $1}"
      shift 2
      ;;
    --marlin-arch=*)
      MARLIN_ARCH="${1#*=}"
      shift
      ;;
    --build-type)
      BUILD_TYPE="${2:?missing value for $1}"
      shift 2
      ;;
    --build-type=*)
      BUILD_TYPE="${1#*=}"
      shift
      ;;
    --build-dir-name)
      BUILD_DIR_NAME="${2:?missing value for $1}"
      shift 2
      ;;
    --build-dir-name=*)
      BUILD_DIR_NAME="${1#*=}"
      shift
      ;;
    --cuda-root)
      CUDA_ROOT="${2:?missing value for $1}"
      shift 2
      ;;
    --cuda-root=*)
      CUDA_ROOT="${1#*=}"
      shift
      ;;
    --ppu-root)
      PPU_ROOT="${2:?missing value for $1}"
      shift 2
      ;;
    --ppu-root=*)
      PPU_ROOT="${1#*=}"
      shift
      ;;
    --cutlass-dir)
      CUTLASS_DIR="${2:?missing value for $1}"
      shift 2
      ;;
    --cutlass-dir=*)
      CUTLASS_DIR="${1#*=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-env-check)
      CHECK_ENV=0
      shift
      ;;
    -v|--verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        TARGET_ARGS+=("$1")
        shift
      done
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      TARGET_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$COMMAND" == "list" ]]; then
  list_targets
  exit 0
fi

if [[ "$COMMAND" == "env" ]]; then
  print_env
  check_environment
  exit 0
fi

if [[ ${#TARGET_ARGS[@]} -eq 0 ]]; then
  TARGET_ARGS=(all)
fi

mapfile -t TARGETS < <(expand_targets "${TARGET_ARGS[@]}")

if [[ "$CHECK_ENV" == 1 && "$COMMAND" != "clean" ]]; then
  check_environment
else
  setup_environment
fi

case "$COMMAND" in
  build)
    for target in "${TARGETS[@]}"; do
      build_target "$target"
    done
    ;;
  configure)
    for target in "${TARGETS[@]}"; do
      configure_target "$target"
    done
    ;;
  clean)
    for target in "${TARGETS[@]}"; do
      clean_target "$target"
    done
    ;;
  rebuild)
    for target in "${TARGETS[@]}"; do
      clean_target "$target"
      build_target "$target"
    done
    ;;
  *)
    die "unknown command: $COMMAND"
    ;;
esac
