/*
# Development Status Notes for Next Team

a key task is GGUF tokenization specification:
the tokenizer is likely inside the .gguf file in some way
but where and how?

## Current Implementation Structure
```rust
src/
  ├── main.rs              // Core model loading and inference
  └── gguf_tokenizer_module/
      └── mod.rs           // GGUF-specific tokenizer implementation
```

## Key Components Documentation

### TokenizerGGUF
```rust
/// GGUF-specific tokenizer implementation
/// 
/// # Design Notes
/// This tokenizer specifically handles the GGUF format's approach to tokenization:
/// - Vocabulary and token mappings stored in model metadata
/// - No external tokenizer files needed (unlike HuggingFace models)
/// - Special tokens (BOS/EOS) stored as explicit IDs in metadata
/// 
/// # Current Status
/// - Basic structure implemented
/// - Vocabulary loading works
/// - Basic encode/decode implemented
/// - NEEDS IMPLEMENTATION: Proper subword tokenization
/// 
/// # Memory Considerations
/// - Vocabulary maps held in memory
/// - Consider memory usage for very large vocabularies
/// 
/// # Future Work Needed
/// 1. Implement proper subword tokenization (current implementation is naive word-splitting)
/// 2. Add proper detokenization (current implementation uses simple space joining)
/// 3. Add handling for special characters and whitespace
/// 4. Add vocabulary statistics and validation
/// 5. Consider adding vocabulary size limits or lazy loading
/// 
/// # Usage Example
/// ```rust
/// let model = GGUFModel::load("model.gguf")?;
/// let tokenizer = model.load_gguf_tokenizer()?;
/// let tokens = tokenizer.encode_text_to_gguf_tokens("Hello world")?;
/// let text = tokenizer.decode_gguf_tokens_to_text(&tokens)?;
/// ```
#[derive(Debug)]
pub struct TokenizerGGUF {
    // ... fields ...
}
```

### Encoding/Decoding Functions
```rust
impl TokenizerGGUF {
    /// Converts input text to GGUF token IDs
    /// 
    /// # Implementation Status
    /// CURRENT: Basic word-splitting only
    /// NEEDED: Proper subword tokenization algorithm
    /// 
    /// # Known Limitations
    /// 1. Only handles space-separated words
    /// 2. No handling of special characters
    /// 3. No handling of case sensitivity
    /// 4. No support for continuous text
    /// 
    /// # Future Requirements
    /// - Implement BPE or WordPiece tokenization
    /// - Add proper text preprocessing
    /// - Handle unknown tokens properly
    pub fn encode_text_to_gguf_tokens(&self, text: &str) -> io::Result<Vec<u32>> {
        // ... implementation ...
    }

    /// Converts GGUF token IDs back to text
    /// 
    /// # Implementation Status
    /// CURRENT: Basic token joining
    /// NEEDED: Proper detokenization rules
    /// 
    /// # Known Limitations
    /// 1. Simple space joining only
    /// 2. No handling of special tokens
    /// 3. No proper whitespace restoration
    /// 
    /// # Future Requirements
    /// - Implement proper detokenization rules
    /// - Handle special tokens correctly
    /// - Restore original whitespace
    pub fn decode_gguf_tokens_to_text(&self, token_ids: &[u32]) -> io::Result<String> {
        // ... implementation ...
    }
}
```

## Critical Next Steps

1. Tokenization Implementation
   - Current tokenization is placeholder only
   - Need to implement proper subword tokenization
   - Reference GGUF specification for exact tokenization rules

2. Memory Management
   - Consider impact of vocabulary size
   - Implement lazy loading if needed
   - Add memory usage tracking

3. Error Handling
   - Add more specific error types
   - Improve error messages
   - Add validation checks

4. Testing
   - Add unit tests for tokenization
   - Add integration tests
   - Add memory usage tests

## Integration Notes

The tokenizer is part of a larger no-load inference implementation:
- Model weights are memory mapped
- Tokenizer vocabulary is currently fully loaded
- Consider impact on overall memory usage

## Usage Warnings

1. Tokenization Implementation
   ```rust
   // WRONG - Will split incorrectly
   let tokens = tokenizer.encode_text_to_gguf_tokens("don't")?;
   
   // TODO: Implement proper subword handling
   // Should handle: contractions, punctuation, special characters
   ```

2. Memory Usage
   ```rust
   // Current implementation loads full vocabulary
   // Consider monitoring memory usage with large models
   let tokenizer = model.load_gguf_tokenizer()?;
   ```

## References Needed
1. GGUF tokenization specification
2. BPE/WordPiece implementation details
3. Memory usage requirements
4. Test suite requirements

notes:
For a LLaMA-style tokenizer (e.g. with 32K vocabulary) the special tokens are:
BOS (begin sequence): ID 1 (<s>)
EOS (end sequence): ID 2 (</s>)
UNK (unknown): ID 0 (<unk>)
PAD (padding): ID 2 (same as EOS)

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

    /// TODO updated Docstring needed!!
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
    
    // pub fn decode_gguf_tokens_to_text(&self, token_ids: &[u32]) -> io::Result<String> {
    //     let mut text = Vec::new();
        
    //     for &token_id in token_ids {
    //         // Skip special tokens
    //         if token_id == self.bos_token || token_id == self.eos_token {
    //             continue;
    //         }
            
    //         if let Some(token) = self.id_to_token.get(&token_id) {
    //             // Handle byte tokens
    //             if token.starts_with("<0x") && token.ends_with('>') {
    //                 // Convert byte token back to actual byte
    //                 if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
    //                     text.push(byte as char);
    //                 }
    //             } else {
    //                 // Regular token
    //                 text.push_str(token);
    //             }
    //         } else {
    //             return Err(io::Error::new(
    //                 io::ErrorKind::InvalidData,
    //                 format!("Token ID not in vocabulary: {}", token_id)
    //             ));
    //         }
    //     }
        
    //     Ok(text.into_iter().collect())
    // }
    
    
    // pub fn encode_text_to_gguf_tokens(&self, text: &str) -> io::Result<Vec<u32>> {
    //     let mut tokens = Vec::new();
        
    //     // Add BOS token
    //     tokens.push(self.bos_token);
        
    //     // TODO: Implement actual tokenization
    //     // This is placeholder - needs proper subword tokenization
    //     for word in text.split_whitespace() {
    //         if let Some(&token_id) = self.vocab.get(word) {
    //             tokens.push(token_id);
    //         } else {
    //             return Err(io::Error::new(
    //                 io::ErrorKind::InvalidInput,
    //                 format!("Token not in GGUF vocabulary: {}", word)
    //             ));
    //         }
    //     }
        
    //     Ok(tokens)
    // }


    // pub fn decode_gguf_tokens_to_text(&self, token_ids: &[u32]) -> io::Result<String> {
    //     let mut text = Vec::new();
        
    //     for &token_id in token_ids {
    //         // Skip special tokens
    //         if token_id == self.bos_token || token_id == self.eos_token {
    //             continue;
    //         }
            
    //         if let Some(token) = self.id_to_token.get(&token_id) {
    //             text.push(token.clone());
    //         } else {
    //             return Err(io::Error::new(
    //                 io::ErrorKind::InvalidData,
    //                 format!("Token ID not in GGUF vocabulary: {}", token_id)
    //             ));
    //         }
    //     }
        
    //     Ok(text.join(" "))  // Simplified joining - needs proper detokenization
    // }
}

/// TODO this needs an updated doc string!
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

