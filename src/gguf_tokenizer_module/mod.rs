/*
# Development Status Notes for Next Team

a key task is GGUF tokenization specification:
the tokenizer is likely inside the .gguf file in some way
but where and how?

# Notes:
## Current Implementation Structure
```rust
src/
  ├── main.rs              // Core model loading and inference
  └── gguf_tokenizer_module/
      └── mod.rs           // GGUF-specific tokenizer implementation
```
For a LLaMA-style tokenizer (e.g. with 32K vocabulary) the special tokens are:
BOS (begin sequence): ID 1 (<s>)
EOS (end sequence): ID 2 (</s>)
UNK (unknown): ID 0 (<unk>)
PAD (padding): ID 2 (same as EOS)

## Documentation
```rust
// gguf_tokenizer_module/mod.rs

/*
# GGUF Tokenizer Module Documentation
Version: 0.1.0
Last Updated: 2024-01-04

## Overview
This module implements tokenization for GGUF format models, specifically handling:
- Vocabulary loading from GGUF metadata
- Token encoding/decoding
- Special token management
- Byte fallback tokenization

## Current Implementation Status

### Working Features:
- Basic metadata extraction
- Special token handling (BOS, EOS, UNK)
- Basic token encoding/decoding
- Byte fallback tokens (<0x00> through <0xFF>)

### Needs Implementation:
1. Proper Subword Tokenization
   - Currently only does basic word splitting
   - Needs BPE or WordPiece implementation
   - Reference: LLaMA tokenizer uses SentencePiece

2. Whitespace Handling
   - Current implementation loses whitespace information
   - Need to properly handle:
     * Leading/trailing whitespace
     * Multiple spaces
     * Special whitespace characters (\n, \t, etc)

3. Performance Optimization
   - Current implementation makes multiple passes over text
   - Need efficient token matching algorithm
   - Consider caching frequent token sequences

## Key Data Structures

### TokenizerGGUF
```rust
pub struct TokenizerGGUF {
    pub vocab: HashMap<String, u32>,        // Token string to ID mapping
    pub id_to_token: HashMap<u32, String>,  // ID to token string mapping
    pub bos_token: u32,                     // Beginning of sequence token ID
    pub eos_token: u32,                     // End of sequence token ID
    pub unknown_token: u32,                 // Unknown token ID
}
```

## GGUF Metadata Fields
Essential fields in model.header.metadata.kvs:
- "tokenizer.ggml.model": String (e.g., "llama")
- "tokenizer.ggml.tokens": Array[String]
- "tokenizer.ggml.bos_token_id": U32
- "tokenizer.ggml.eos_token_id": U32
- "tokenizer.ggml.unknown_token_id": U32
- "tokenizer.ggml.padding_token_id": U32

## Usage Example
```rust
let model = GGUFModel::load("model.gguf")?;
let tokenizer = model.load_tokenizer()?;
let tokens = tokenizer.encode_text_to_gguf_tokens("Hello world")?;
let text = tokenizer.decode_gguf_tokens_to_text(&tokens)?;
```

## Critical Next Steps

1. Implement Proper Subword Tokenization:
```rust
// TODO: Implement proper subword tokenization
fn tokenize_subword(&self, text: &str) -> Vec<String> {
    // Need to implement:
    // 1. Proper token matching algorithm
    // 2. Longest-match-first strategy
    // 3. Handle overlapping tokens
}
```

2. Add Proper Whitespace Handling:
```rust
// TODO: Implement whitespace preservation
fn preserve_whitespace(&self, text: &str) -> Vec<(String, bool)> {
    // Need to:
    // 1. Track whitespace locations
    // 2. Preserve in token sequence
    // 3. Restore during decoding
}
```

3. Implement Token Sequence Building:
```rust
// TODO: Implement efficient token sequence building
fn build_token_sequence(&self, tokens: Vec<String>) -> Vec<u32> {
    // Need to:
    // 1. Handle special tokens
    // 2. Apply vocabulary efficiently
    // 3. Use byte fallback when needed
}
```

## Memory Considerations
- Current vocabulary storage: O(V) where V is vocabulary size
- Token sequence storage: O(N) where N is sequence length
- Consider streaming for long sequences
- May need memory-mapped vocabulary for very large models

## Error Handling
Current error cases:
- Missing metadata fields
- Invalid token IDs
- Unknown tokens
Need to add:
- Better error messages
- Recovery strategies
- Validation checks

## Testing Needed
1. Unit Tests:
   - Special token handling
   - Byte fallback cases
   - Whitespace preservation
   - Edge cases (empty string, all special chars, etc)

2. Integration Tests:
   - Full encode-decode cycle
   - Long text sequences
   - Memory usage patterns
   - Performance benchmarks

## References
1. GGUF Format Specification:
   https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
2. LLaMA Tokenizer Details:
   [Need to add reference to specific LLaMA tokenizer documentation]
3. SentencePiece Implementation:
   https://github.com/google/sentencepiece

## Known Issues
1. Token sequence length not properly handled
2. Whitespace lost in encoding/decoding
3. No proper subword tokenization
4. Performance not optimized for large texts

## Future Improvements
1. Streaming token processing
2. Cached token sequences
3. Memory-mapped vocabulary
4. Better error recovery
5. Performance optimizations

## Contact
[Add team contact information here]
*/
```

*/

// name of module: gguf_tokenizer_module

use std::collections::HashMap;
use std::io;
use crate::{GGUFModel, MetadataValue};  // Note: using crate:: to reference main module

/// General tokenizer interface, format-agnostic
#[derive(Debug)]
pub struct TokenizerGGUF {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub bos_token: u32,
    pub eos_token: u32,
    pub unknown_token: u32, 
}

/// TokenizerGGUF struct for gguf_tokenizer_module/mod.rs
impl TokenizerGGUF {
    
    
    /// Converts input text to GGUF token IDs
    /// 
    /// # Design Notes
    /// - Adds BOS token at start
    /// - Splits text into subwords using GGUF vocabulary
    /// - Returns vector of token IDs
    /// 
    /// # Arguments
    /// * `text` - Input text to tokenize
    /// 
    /// # Returns
    /// * `io::Result<Vec<u32>>` - Vector of token IDs including BOS token
    /// 
    /// # Errors
    /// * `io::Error` if text contains tokens not in vocabulary
    /// Converts input text to GGUF token IDs
    pub fn encode_text_to_gguf_tokens(&self, text: &str) -> io::Result<Vec<u32>> {
        let mut tokens = Vec::new();
        
        // Add BOS token
        tokens.push(self.bos_token);
        
        // Basic word splitting (this needs to be improved with proper subword tokenization)
        for word in text.split_whitespace() {
            if let Some(&token_id) = self.vocab.get(word) {
                tokens.push(token_id);
            } else {
                // If word not found, try byte fallback
                for byte in word.bytes() {
                    let byte_token = format!("<0x{:02X}>", byte);
                    if let Some(&token_id) = self.vocab.get(&byte_token) {
                        tokens.push(token_id);
                    } else {
                        tokens.push(self.unknown_token);
                    }
                }
            }
        }
        
        // Add EOS token
        tokens.push(self.eos_token);
        
        Ok(tokens)
    }

    // TODO updated Docstring needed!!
    /// Converts GGUF token IDs back to text
    /// Converts GGUF token IDs back to text
    /// 
    /// # Design Notes
    /// - Skips special tokens (BOS/EOS)
    /// - Looks up each token ID in vocabulary
    /// - Joins tokens with spaces (simplified)
    /// 
    /// # Arguments
    /// * `token_ids` - Vector of GGUF token IDs
    /// 
    /// # Returns
    /// * `io::Result<String>` - Decoded text
    /// 
    /// # Errors
    /// * `io::Error` if any token ID not found in vocabulary
    /// Converts GGUF token IDs back to text
    pub fn decode_gguf_tokens_to_text(&self, token_ids: &[u32]) -> io::Result<String> {
        let mut text = String::new();  // Changed from Vec<char> to String
        
        for &token_id in token_ids {
            // Skip special tokens
            if token_id == self.bos_token || token_id == self.eos_token {
                continue;
            }
            
            if let Some(token) = self.id_to_token.get(&token_id) {
                // Handle byte tokens
                if token.starts_with("<0x") && token.ends_with('>') {
                    // Convert byte token back to actual byte
                    if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                        text.push(byte as char);
                    }
                } else {
                    // Regular token
                    text.push_str(token);  // Now this will work with String
                }
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Token ID not in vocabulary: {}", token_id)
                ));
            }
        }
        
        Ok(text)
    }
} // end imp TOKENIZERGGUF

// TODO this needs an updated doc string!
/// Loads tokenizer data specifically from GGUF format metadata
/// 
/// # Design Notes
/// GGUF stores tokenizer data differently from other formats:
/// - Vocabulary is stored in metadata under "tokenizer.ggml.tokens"
/// - Special tokens are stored as IDs: "tokenizer.ggml.bos_token_id", etc
/// - No external files needed (unlike HuggingFace which uses tokenizer.json)
pub fn load_from_gguf(model: &GGUFModel) -> io::Result<TokenizerGGUF> {
    // Extract vocabulary from GGUF metadata
    let vocab_tokens = match &model.header.metadata.kvs.get("tokenizer.ggml.tokens") {
        Some(MetadataValue::Array(tokens)) => tokens,
        _ => return Err(io::Error::new(io::ErrorKind::NotFound, "GGUF tokenizer vocab not found")),
    };
    
    // Build mappings
    let mut vocab = HashMap::new();
    let mut id_to_token = HashMap::new();
    
    for (i, token) in vocab_tokens.iter().enumerate() {
        if let MetadataValue::String(s) = token {
            vocab.insert(s.data.clone(), i as u32);
            id_to_token.insert(i as u32, s.data.clone());
        }
    }
    
    // Get GGUF special token IDs
    let bos_token = match &model.header.metadata.kvs.get("tokenizer.ggml.bos_token_id") {
        Some(MetadataValue::U32(id)) => *id,
        _ => return Err(io::Error::new(io::ErrorKind::NotFound, "GGUF BOS token ID not found")),
    };
    
    let eos_token = match &model.header.metadata.kvs.get("tokenizer.ggml.eos_token_id") {
        Some(MetadataValue::U32(id)) => *id,
        _ => return Err(io::Error::new(io::ErrorKind::NotFound, "GGUF EOS token ID not found")),
    };

    let unknown_token = match &model.header.metadata.kvs.get("tokenizer.ggml.unknown_token_id") {
        Some(MetadataValue::U32(id)) => *id,
        _ => return Err(io::Error::new(io::ErrorKind::NotFound, "GGUF unknown token ID not found")),
    };
    
    Ok(TokenizerGGUF {
        vocab,
        id_to_token,
        bos_token,
        eos_token,
        unknown_token,
    })
}

// pub fn load_from_gguf(model: &GGUFModel) -> io::Result<TokenizerGGUF> {
//     // Extract vocabulary from GGUF metadata
//     let vocab_tokens = match &model.header.metadata.kvs.get("tokenizer.ggml.tokens") {
//         Some(MetadataValue::Array(tokens)) => tokens,
//         _ => return Err(io::Error::new(io::ErrorKind::NotFound, "GGUF tokenizer vocab not found")),
//     };
    
//     // Build mappings
//     let mut vocab = HashMap::new();
//     let mut id_to_token = HashMap::new();
    
//     for (i, token) in vocab_tokens.iter().enumerate() {
//         if let MetadataValue::String(s) = token {
//             vocab.insert(s.data.clone(), i as u32);
//             id_to_token.insert(i as u32, s.data.clone());
//         }
//     }
    
//     // Get GGUF special token IDs
//     let bos_token = match &model.header.metadata.kvs.get("tokenizer.ggml.bos_token_id") {
//         Some(MetadataValue::U32(id)) => *id,
//         _ => return Err(io::Error::new(io::ErrorKind::NotFound, "GGUF BOS token ID not found")),
//     };
    
//     let eos_token = match &model.header.metadata.kvs.get("tokenizer.ggml.eos_token_id") {
//         Some(MetadataValue::U32(id)) => *id,
//         _ => return Err(io::Error::new(io::ErrorKind::NotFound, "GGUF EOS token ID not found")),
//     };
    
//     Ok(TokenizerGGUF {
//         vocab,
//         id_to_token,
//         bos_token,
//         eos_token,
//     })
// }

