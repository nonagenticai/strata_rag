# strata-rag, CPU-only Linux image. strata runs no model: embeddings and reranking come from the
# HTTP providers (providers.toml / tenant registrations), so this image is the full service.
#
# Base images are debian:13-slim from Docker Hub. There is no glibc Rust image
# (library/rust is 1-alpine only), so the builder is debian:13-slim with a
# pinned Rust toolchain installed by rustup. That keeps builder and runtime on the same
# glibc, which the tokenizers (C++ esaxx, onig) / aws-lc native builds expect.
# Pin both FROM lines by digest when promoting this into a pipeline.
ARG RUST_TOOLCHAIN=1.100.0

FROM debian:13-slim AS build
ARG RUST_TOOLCHAIN
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl perl pkg-config \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" \
    && rustc --version
# bge-m3's tokenizer.json (chunk token counting; no weights) is baked in at a pinned revision and
# checksum, so the pod needs no Hugging Face egress, no cache volume and no writable root filesystem.
# (DL3064 below fires on the substring TOKEN; none of these is a secret.)
# The revision must match REVISION in src/text/tokenizer.rs (the native path's hf-hub lookup).
# hadolint ignore=DL3064
ARG TOKENIZER_REVISION=038773d84178649586cfdf228df3e2a0838a5180
# hadolint ignore=DL3064
ARG TOKENIZER_SHA256=21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08
RUN curl --proto '=https' --tlsv1.2 -fsSL --retry 3 \
        --create-dirs -o /out/tokenizer.json \
        "https://huggingface.co/BAAI/bge-m3/resolve/${TOKENIZER_REVISION}/tokenizer.json" \
    && echo "${TOKENIZER_SHA256}  /out/tokenizer.json" | sha256sum -c -


WORKDIR /src
COPY . .
# sqlx::migrate! embeds migrations/ into the binary, so the runtime image needs no SQL files.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked \
    && install -D -m 0755 target/release/strata /out/strata

FROM debian:13-slim AS runtime
# ca-certificates: rustls-native-certs (providers / laya over https). curl: the HEALTHCHECK.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 65532 strata \
    && useradd --system --uid 65532 --gid 65532 --home-dir /var/lib/strata --create-home strata
COPY --from=build /out/strata /usr/local/bin/strata
COPY --from=build /out/tokenizer.json /usr/share/strata/tokenizer.json
# STRATA_TOKENIZER_PATH makes strata read the baked tokenizer and never touch hf-hub.
# The provider base URLs have no default and must be set (startup fails naming the missing one).
# DL3064 matches "TOKEN" in TOKENIZER; these are a revision, a checksum and a file path, not secrets.
# hadolint ignore=DL3064
ENV STRATA_BIND=0.0.0.0:8092 \
    STRATA_TOKENIZER_PATH=/usr/share/strata/tokenizer.json \
    RUST_LOG=info
WORKDIR /var/lib/strata
USER 65532:65532
EXPOSE 8092
# /health answers 200 with a JSON component report even when a dependency is down, so this is a
# liveness check of the HTTP server; read the body for db/embed/rerank/chat/laya status.
# start-period covers DB connect + migrations.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["curl", "-fsS", "-o", "/dev/null", "http://127.0.0.1:8092/health"]
ENTRYPOINT ["/usr/local/bin/strata"]
CMD ["serve"]
