//! Text preparation: normalisation and token-aware chunking.

pub mod chunk;
pub mod normalize;
pub mod tokenizer;

pub use chunk::{Chunk, HfTokenCounter, TokenCounter, WhitespaceCounter, chunk_text};
pub use normalize::normalize;
pub use tokenizer::bge_m3_tokenizer_path;
