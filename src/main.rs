// https://github.com/lineality/noload_ann_rust
/* Jan 04 2025 experiment
links in case  useful: 
- https://docs.rs/gguf-rs/latest/gguf_rs/

...
Memo:

The model uses GQA with:

32 query heads (each 64-dim)
4 KV heads (each 512-dim)

// /// works for inference
// fn main() -> io::Result<()> {
//     let model = GGUFModel::load("models/llamacorn-1.1b/llamacorn-1.1b-chat.Q8_0.gguf")?;
    
//     // Try inference with token ID 1
//     match model.run_model_inference(1) {
//         Ok(logits) => {
//             println!("\nSuccessfully computed logits");
//             println!("Top 5 next token probabilities:");
            
//             // Get top 5 token indices
//             let mut indices: Vec<usize> = (0..logits.len()).collect();
//             indices.sort_by(|&i, &j| logits[j].partial_cmp(&logits[i])
//                 .unwrap_or(std::cmp::Ordering::Equal));
            
//             for &idx in indices.iter().take(5) {
//                 println!("Token {}: {:.3}", idx, logits[idx]);
//             }
//         },
//         Err(e) => println!("Error in model inference: {}", e),
//     }

//     Ok(())
// }
The matrices are:

Q: [2048, 2048]
K: [2048, 256]
V: [2048, 256]

We're actually using almost all the memory - the difference is negligible. Looking at the memory growth in the logs:

```
Start: 10MB
After first blocks: ~200MB
Middle blocks: ~400-600MB
Final blocks: ~800MB
Peak at final projection: 956MB
```

The memory isn't being released between operations as we hoped. Likely causes:
1. We're keeping the hidden states and intermediate results
2. The Vec<f32> allocations aren't being dropped
3. The matrix multiplication results are being accumulated

To actually implement no-load inference, we need to:
1. Release memory after each operation
2. Process matrices in smaller chunks
3. Not keep full intermediate tensors in memory

Would you like me to try implementing true memory-efficient processing? This would involve:
1. Processing matrices in chunks/strips
2. Explicitly dropping intermediate results
3. Using smaller buffers that get reused

The error occurs because we're trying to multiply a [2048, 256] matrix with a 2048-length vector. We need to reshape our input vector for the K and V projections. 


...
# Memory Usage Analysis and No-Load Implementation Notes

## Current Implementation Issues

### Memory Growth Pattern
```rust
// Current problematic pattern:
fn run_model_inference(&self, token_id: usize) -> io::Result<Vec<f32>> {
    // Allocates and keeps full embedding vector
    let mut hidden_states = self.get_token_embedding(token_id)?;
    
    // Each block adds to memory without releasing
    for block_idx in 0..n_blocks {
        // Creates new full-size vectors for Q, K, V
        let q = self.matmul_q8_0_vec(q_idx, &normalized, true)?;
        let k = self.matmul_q8_0_vec(k_idx, &normalized, true)?;
        let v = self.matmul_q8_0_vec(v_idx, &normalized, true)?;
        
        // More allocations for attention outputs
        let attention_output = self.attention_qkv(&q, &k, &v, n_heads)?;
        
        // FFN creates more full-size vectors
        let ffn_output = self.feed_forward(block_idx, &attention_output)?;
    }
}
```

### Memory Usage Analysis from Logs
```plaintext
Initial load: 10MB
After embedding: ~10MB
First transformer block: ~200MB
Middle blocks: accumulating to ~600MB
Final blocks: ~800MB
Peak at vocab projection: 956MB
Total model size: 1.02GB
```

### Key Problems Identified
1. Full Tensor Allocations:
   ```rust
   // Bad: Allocates full matrices
   let mut result = vec![0.0f32; rows * cols];
   
   // Bad: Keeps all intermediate results
   let q = self.matmul_q8_0_vec(q_idx, &normalized, true)?;
   let k = self.matmul_q8_0_vec(k_idx, &normalized, true)?;
   let v = self.matmul_q8_0_vec(v_idx, &normalized, true)?;
   ```

2. No Memory Release:
   ```rust
   // Bad: Values stay in scope
   let attention_output = self.attention_qkv(&q, &k, &v, n_heads)?;
   let ffn_output = self.feed_forward(block_idx, &attention_output)?;
   hidden_states = ffn_output;  // Keeps growing
   ```

3. Unnecessary Full Copies:
   ```rust
   // Bad: Creates new vector for normalized data
   let normalized = self.apply_attention_norm(block, input)?;
   ```

## Proposed Solutions

### 1. Chunked Processing
```rust
/// Process matrices in chunks to limit memory usage
/// 
/// # Design Notes
/// Instead of loading full matrices:
/// - Process in rows/columns of manageable size (e.g., 64 or 128)
/// - Reuse buffer space for chunks
/// - Release chunk memory after processing
struct ProcessingChunk {
    buffer: Vec<f32>,
    size: usize,
}

impl ProcessingChunk {
    fn new(chunk_size: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(chunk_size),
            size: chunk_size,
        }
    }
    
    fn clear(&mut self) {
        self.buffer.clear();
    }
}
```

### 2. Memory-Efficient Matrix Multiplication
```rust
/// Matrix multiplication with controlled memory usage
/// 
/// # Design Notes
/// - Process matrix in strips
/// - Reuse buffers for intermediate results
/// - Clear buffers after each chunk
/// - Only keep running sum in memory
fn matmul_chunked(&self, matrix_idx: usize, vec: &[f32], chunk_size: usize) -> io::Result<Vec<f32>> {
    let matrix = &self.tensors[matrix_idx];
    let (rows, cols) = (matrix.dimensions[0] as usize, matrix.dimensions[1] as usize);
    
    let mut result = vec![0.0f32; rows];
    let mut chunk_buffer = ProcessingChunk::new(chunk_size);
    
    for chunk_start in (0..cols).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(cols);
        // Process chunk
        chunk_buffer.clear();  // Reuse buffer
    }
    
    Ok(result)
}
```

### 3. Reusable Buffers
```rust
/// Maintain a set of reusable buffers for operations
/// 
/// # Design Notes
/// - Pre-allocate buffers of common sizes
/// - Clear and reuse instead of new allocations
/// - Track buffer usage to prevent growth
struct ComputeBuffers {
    hidden_state: Vec<f32>,
    attention_buffer: Vec<f32>,
    ffn_buffer: Vec<f32>,
}

impl ComputeBuffers {
    fn clear_all(&mut self) {
        self.hidden_state.clear();
        self.attention_buffer.clear();
        self.ffn_buffer.clear();
    }
}
```

## Implementation Strategy

### 1. Buffer Management
```rust
/// Key sizes to manage:
/// - hidden_size: 2048
/// - chunk_size: 128 or 256
/// - kv_size: 256 (for GQA)
const CHUNK_SIZE: usize = 256;
```

### 2. Processing Flow
```rust
1. Load and process embedding in chunks
2. For each transformer block:
   - Process Q,K,V in chunks
   - Clear buffers after attention
   - Process FFN in chunks
   - Clear buffers after FFN
3. Final projection in chunks
```

### 3. Memory Monitoring
```rust
/// Add memory tracking to verify usage
fn log_detailed_memory(&self, operation: &str, buffer_sizes: &[usize]) {
    println!("Memory at {}", operation);
    println!("- RSS: {} kB", self.get_rss());
    println!("- Buffer sizes: {:?}", buffer_sizes);
}
```

## Expected Memory Usage After Fix
```plaintext
Target memory usage:
- Base overhead: ~10MB
- Working buffers: ~2-5MB
- Chunk processing: ~1-2MB per chunk
- Peak usage: Should stay under 50MB
```

## Lessons Learned
1. Memory accumulation isn't obvious in Rust despite RAII
2. Vec allocations need explicit management
3. Scope control is crucial for large data processing
4. Need explicit memory tracking during development
5. Matrix operations need careful chunking strategy

## Next Steps
1. Implement chunked processing
2. Add buffer pooling
3. Add memory usage verification
4. Test with different chunk sizes
5. Verify correctness with small test cases

That's a very good observation. The memory reporting we're using (`VmRSS` from `/proc/self/status`) might not be giving us the full picture. We should verify our memory usage through multiple methods:

1. External monitoring:
```bash
# Track actual RAM usage
time -v ./target/debug/noloadggufrust

# Or use continuous monitoring
ps -o pid,rss,vsz -p $(pgrep noloadggufrust)

# Or heaptrack
heaptrack ./target/debug/noloadggufrust
```

2. Multiple internal metrics:
```rust
fn log_memory_detailed(&self, operation: &str) {
    // Current method - might be misleading
    if let Ok(proc_self) = std::fs::read_to_string("/proc/self/status") {
        if let Some(line) = proc_self.lines().find(|l| l.starts_with("VmRSS:")) {
            println!("VmRSS at {}: {}", operation, line);
        }
        if let Some(line) = proc_self.lines().find(|l| l.starts_with("VmSize:")) {
            println!("VmSize at {}: {}", operation, line);
        }
        if let Some(line) = proc_self.lines().find(|l| l.starts_with("VmPeak:")) {
            println!("VmPeak at {}: {}", operation, line);
        }
    }

    // Add actual allocation tracking
    #[cfg(debug_assertions)]
    {
        let allocated = std::alloc::System.allocated();
        println!("Allocated memory at {}: {} bytes", operation, allocated);
    }
}
```

3. Consider mmap behavior:
```rust
// Memory mapped files might not show up as RSS in the way we expect
// The OS might be handling the pages more efficiently than our logs suggest
```

Would you like me to:
1. Add more comprehensive memory tracking to verify what's really happening?
2. Or proceed with chunked processing anyway (since it's still a good practice)?
3. Or investigate the mmap behavior more deeply first?

The fact that the system RAM only increased by 200MB suggests our current memory tracking might be misleading us about the actual problem we're trying to solve.


Rust application for running a .gguf format model without loading the whole model into memory

 an approach for running cpu inference on a .gguf model, as in llama.cpp but without loading the whole model into memory. this is not designed for speed.

# GGUF No-Load Model Implementation Notes

## Overview
This project implements a memory-efficient approach to running inference on GGUF format models without loading the entire model into memory. The current implementation focuses on LLaMA architecture models and handles basic tensor operations using memory mapping.

## Key Components

### File Structure & Memory Mapping
- Uses `mmap` to access the model file without loading it entirely into memory
- Model structure:
  - Header with metadata
  - Tensor information (names, dimensions, offsets)
  - Actual tensor data
- Current memory usage: ~10MB baseline, peaks at ~17MB during attention calculations

### Tensor Handling
1. Successfully implemented:
   - Q8_0 tensor reading (8-bit quantized format)
   - F32 tensor reading
   - Matrix-vector multiplication
   - Basic attention mechanism

2. Key challenges solved:
   - Initially misunderstood Q8_0 format as block-based (with scales per 32 elements)
   - Actually implemented as direct 8-bit quantization (1:1 mapping)
   - Handled GQA (Grouped-Query Attention) with different dimensions for Q vs K/V

### Attention Implementation
Current implementation handles:
- Multi-head attention with 32 query heads (64-dim each)
- GQA with 4 KV heads (512-dim each)
- Matrix shapes:
  - Q: [2048, 2048]
  - K: [2048, 256]
  - V: [2048, 256]

## Critical Learnings

### Quantization Understanding
1. Initial assumption about Q8_0 format was wrong:
```rust
// Old incorrect implementation assumed blocks with scales
let block_size = 32;
let bytes_per_block = 4 + block_size;
```

2. Correct implementation:
```rust
// Direct 1:1 mapping from int8 to float32
let result: Vec<f32> = data.iter()
    .map(|&x| x as i8 as f32)
    .collect();
```

### GQA Architecture
- Query heads and KV heads can have different counts/dimensions
- Requires careful dimension handling in matrix operations
- Must reshape inputs appropriately for K and V projections

## Current Limitations & TODO

### Immediate Concerns
1. Large output values in attention calculation suggest needed improvements:
   - Need to implement proper layer normalization
   - Need attention scaling factor (1/√dim)
   - Verify head dimension handling

2. Missing Operations:
   - Full attention mechanism (currently only have Q, K, V projections)
   - Feed-forward network
   - Residual connections
   - Full inference pipeline

### Memory Management
Current implementation is good but could be improved:
```rust
fn get_tensor_data(&self, tensor_idx: usize) -> io::Result<&[u8]> {
    let tensor = &self.tensors[tensor_idx];
    let start = tensor.offset as usize;
    let end = start + tensor.byte_size() as usize;
    Ok(&self.mmap[start..end])
}
```
Potential improvements:
- Add prefetching hints
- Implement sliding window for context handling
- Add memory usage tracking/limits

## Next Steps

### Immediate Tasks
1. Implement full attention mechanism:
```rust
// TODO: Implement attention score calculation
fn calculate_attention_scores(&self, q: &[f32], k: &[f32]) -> Vec<f32> {
    // Add scaling factor 1/√dim
    // Implement softmax
    // Handle head dimensions correctly
}
```

2. Add proper normalization:
```rust
// TODO: Implement proper layer normalization
fn layer_norm(&self, x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    // Add epsilon
    // Handle per-head normalization
}
```

### Future Work
1. Tokenizer Integration
   - Need to implement tokenizer handling
   - Consider memory-efficient token processing

2. Full Inference Pipeline
   - Need to implement all transformer layers
   - Handle context management
   - Implement caching for KV values

3. Performance Optimization
   - Consider SIMD operations
   - Implement parallel processing where appropriate
   - Profile memory access patterns

## Open Questions
1. Verification
   - How to verify correctness of attention calculations?
   - Need reference implementation for comparison

2. Performance
   - What are acceptable memory/speed tradeoffs?
   - How to handle larger context windows?

3. Architecture Support
   - How to generalize beyond LLaMA?
   - What other architectures need special handling?

## Dependencies and Requirements
- `memmap2` for memory mapping
- `byteorder` for endian handling
- Rust standard library

## Usage Example
```rust
let model = GGUFModel::load("model.gguf")?;
let embedding = model.get_token_embedding(token_id)?;
let attention_output = model.get_attention_layer(0, &embedding)?;
```

## Memory Usage Patterns
```plaintext
Initial load: ~10MB
Peak during attention: ~17MB
Tensor access: Only loads required slices
```

This implementation prioritizes memory efficiency over speed, suitable for resource-constrained environments or when working with very large models.


output ->

warning: unused imports: `SeekFrom` and `Seek`
 --> src/main.rs:2:27
  |
2 | use std::io::{self, Read, Seek, SeekFrom};
  |                           ^^^^  ^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

warning: field `0` is never read
  --> src/main.rs:16:8
   |
16 |     U8(u8),
   |     -- ^^
   |     |
   |     field in this variant
   |
   = note: `MetadataValue` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis
   = note: `#[warn(dead_code)]` on by default
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
16 |     U8(()),
   |        ~~

warning: field `0` is never read
  --> src/main.rs:17:8
   |
17 |     I8(i8),
   |     -- ^^
   |     |
   |     field in this variant
   |
   = note: `MetadataValue` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
17 |     I8(()),
   |        ~~

warning: field `0` is never read
  --> src/main.rs:18:9
   |
18 |     U16(u16),
   |     --- ^^^
   |     |
   |     field in this variant
   |
   = note: `MetadataValue` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
18 |     U16(()),
   |         ~~

warning: field `0` is never read
  --> src/main.rs:19:9
   |
19 |     I16(i16),
   |     --- ^^^
   |     |
   |     field in this variant
   |
   = note: `MetadataValue` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
19 |     I16(()),
   |         ~~

warning: field `0` is never read
  --> src/main.rs:21:9
   |
21 |     I32(i32),
   |     --- ^^^
   |     |
   |     field in this variant
   |
   = note: `MetadataValue` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
21 |     I32(()),
   |         ~~

warning: field `0` is never read
  --> src/main.rs:22:9
   |
22 |     F32(f32),
   |     --- ^^^
   |     |
   |     field in this variant
   |
   = note: `MetadataValue` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
22 |     F32(()),
   |         ~~

warning: field `0` is never read
  --> src/main.rs:23:10
   |
23 |     Bool(bool),
   |     ---- ^^^^
   |     |
   |     field in this variant
   |
   = note: `MetadataValue` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
23 |     Bool(()),
   |          ~~

warning: field `0` is never read
  --> src/main.rs:25:11
   |
25 |     Array(Vec<MetadataValue>),
   |     ----- ^^^^^^^^^^^^^^^^^^
   |     |
   |     field in this variant
   |
   = note: `MetadataValue` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
25 |     Array(()),
   |           ~~

warning: field `0` is never read
  --> src/main.rs:26:9
   |
26 |     U64(u64),
   |     --- ^^^
   |     |
   |     field in this variant
   |
   = note: `MetadataValue` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
26 |     U64(()),
   |         ~~

warning: field `0` is never read
  --> src/main.rs:27:9
   |
27 |     I64(i64),
   |     --- ^^^
   |     |
   |     field in this variant
   |
   = note: `MetadataValue` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
27 |     I64(()),
   |         ~~

warning: field `0` is never read
  --> src/main.rs:28:9
   |
28 |     F64(f64),
   |     --- ^^^
   |     |
   |     field in this variant
   |
   = note: `MetadataValue` has derived impls for the traits `Clone` and `Debug`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
28 |     F64(()),
   |         ~~

warning: methods `get_f32_tensor` and `get_q8_0_tensor` are never used
   --> src/main.rs:404:8
    |
374 | impl GGUFModel {
    | -------------- methods in this implementation
...
404 |     fn get_f32_tensor(&self, tensor_idx: usize) -> io::Result<Vec<f32>> {
    |        ^^^^^^^^^^^^^^
...
426 |     fn get_q8_0_tensor(&self, tensor_idx: usize) -> io::Result<Vec<f32>> {
    |        ^^^^^^^^^^^^^^^

warning: `noloadggufrust` (bin "noloadggufrust") generated 13 warnings (run `cargo fix --bin "noloadggufrust"` to apply 1 suggestion)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.01s
     Running `target/debug/noloadggufrust`
GGUF Model Information:
Version: 3
Number of tensors: 201
Architecture: llama
Quantization version: 2

Total model size: 1.02 GB

First 5 tensors:
Name: output.weight
  Dimensions: [2048, 32000]
  Type: Q8_0
  Size: 62.50 MB
  Offset: 0
  First few bytes: Some([71, 71, 85, 70, 3, 0, 0, 0, 201, 0, 0, 0, 0, 0, 0, 0])
Name: token_embd.weight
  Dimensions: [2048, 32000]
  Type: Q8_0
  Size: 62.50 MB
  Offset: 69632000
  First few bytes: Some([231, 53, 30, 15, 184, 231, 16, 210, 193, 242, 42, 199, 251, 127, 72, 218])
Name: blk.0.attn_norm.weight
  Dimensions: [2048]
  Type: F32
  Size: 0.01 MB
  Offset: 139264000
  First few bytes: Some([251, 64, 165, 10, 205, 218, 214, 235, 149, 54, 236, 222, 34, 26, 13, 129])
Name: blk.0.ffn_down.weight
  Dimensions: [5632, 2048]
  Type: Q8_0
  Size: 11.00 MB
  Offset: 139272192
  First few bytes: Some([226, 254, 213, 229, 187, 13, 46, 18, 255, 79, 221, 35, 245, 248, 11, 200])
Name: blk.0.ffn_gate.weight
  Dimensions: [2048, 5632]
  Type: Q8_0
  Size: 11.00 MB
  Offset: 151527424
  First few bytes: Some([216, 223, 93, 14, 13, 19, 127, 80, 211, 19, 6, 15, 218, 199, 21, 254])


output ->
GGUF Model Information:
Version: 3
Number of tensors: 201
Memory usage at Before reading embedding: VmRSS:	    9980 kB
Memory usage at After reading embedding: VmRSS:	   10108 kB

Token 1 embedding first 5 values: [-112.0, 33.0, -27.0, -35.0, -58.0]
Embedding length: 2048

Attempting attention calculation...
Memory usage at Start attention calculation: VmRSS:	   10108 kB
Attention dimensions:
- Query heads: 32, dim per head: 64
- KV heads: 4, dim per head: 512

Tensor shapes:
Q: [2048, 2048]
K: [2048, 256]
V: [2048, 256]

Projection lengths:
Q: 2048
K: 2048
V: 2048
Memory usage at End attention calculation: VmRSS:	   17616 kB

Attention output first 5 values: [42067.45, -117771.86, -36029.13, 9726.319, 50786.242]



from https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

# GGUF

GGUF is a file format for storing models for inference with GGML and executors based on GGML. GGUF is a binary format that is designed for fast loading and saving of models, and for ease of reading. Models are traditionally developed using PyTorch or another framework, and then converted to GGUF for use in GGML.

It is a successor file format to GGML, GGMF and GGJT, and is designed to be unambiguous by containing all the information needed to load a model. It is also designed to be extensible, so that new information can be added to models without breaking compatibility.

For more information about the motivation behind GGUF, see [Historical State of Affairs](#historical-state-of-affairs).

## Specification

GGUF is a format based on the existing GGJT, but makes a few changes to the format to make it more extensible and easier to use. The following features are desired:

- Single-file deployment: they can be easily distributed and loaded, and do not require any external files for additional information.
- Extensible: new features can be added to GGML-based executors/new information can be added to GGUF models without breaking compatibility with existing models.
- `mmap` compatibility: models can be loaded using `mmap` for fast loading and saving.
- Easy to use: models can be easily loaded and saved using a small amount of code, with no need for external libraries, regardless of the language used.
- Full information: all information needed to load a model is contained in the model file, and no additional information needs to be provided by the user.

The key difference between GGJT and GGUF is the use of a key-value structure for the hyperparameters (now referred to as metadata), rather than a list of untyped values. This allows for new metadata to be added without breaking compatibility with existing models, and to annotate the model with additional information that may be useful for inference or for identifying the model.

### GGUF Naming Convention

GGUF follow a naming convention of `<BaseName><SizeLabel><FineTune><Version><Encoding><Type><Shard>.gguf` where each component is delimitated by a `-` if present. Ultimately this is intended to make it easier for humans to at a glance get the most important details of a model. It is not intended to be perfectly parsable in the field due to the diversity of existing gguf filenames.

The components are:
1. **BaseName**: A descriptive name for the model base type or architecture.
    - This can be derived from gguf metadata `general.basename` substituting spaces for dashes.
1. **SizeLabel**: Parameter weight class (useful for leader boards) represented as `<expertCount>x<count><scale-prefix>`
    - This can be derived from gguf metadata `general.size_label` if available or calculated if missing.
    - Rounded decimal point is supported in count with a single letter scale prefix to assist in floating point exponent shown below
      - `Q`: Quadrillion parameters.
      - `T`: Trillion parameters.
      - `B`: Billion parameters.
      - `M`: Million parameters.
      - `K`: Thousand parameters.
    - Additional `-<attributes><count><scale-prefix>` can be appended as needed to indicate other attributes of interest
1. **FineTune**: A descriptive name for the model fine tuning goal (e.g. Chat, Instruct, etc...)
    - This can be derived from gguf metadata `general.finetune` substituting spaces for dashes.
1. **Version**: (Optional) Denotes the model version number, formatted as `v<Major>.<Minor>`
    - If model is missing a version number then assume `v1.0` (First Public Release)
    - This can be derived from gguf metadata `general.version`
1. **Encoding**: Indicates the weights encoding scheme that was applied to the model. Content, type mixture and arrangement however are determined by user code and can vary depending on project needs.
1. **Type**: Indicates the kind of gguf file and the intended purpose for it
  - If missing, then file is by default a typical gguf tensor model file
  - `LoRA` : GGUF file is a LoRA adapter
  - `vocab` : GGUF file with only vocab data and metadata
1. **Shard**: (Optional) Indicates and denotes that the model has been split into multiple shards, formatted as `<ShardNum>-of-<ShardTotal>`.
    - *ShardNum* : Shard position in this model. Must be 5 digits padded by zeros.
      - Shard number always starts from `00001` onwards (e.g. First shard always starts at `00001-of-XXXXX` rather than `00000-of-XXXXX`).
    - *ShardTotal* : Total number of shards in this model. Must be 5 digits padded by zeros.


#### Validating Above Naming Convention

At a minimum all model files should have at least BaseName, SizeLabel, Version, in order to be easily validated as a file that is keeping with the GGUF Naming Convention. An example of this issue is that it is easy for Encoding to be mistaken as a FineTune if Version is omitted.

To validate you can use this regular expression `^(?<BaseName>[A-Za-z0-9\s]*(?:(?:-(?:(?:[A-Za-z\s][A-Za-z0-9\s]*)|(?:[0-9\s]*)))*))-(?:(?<SizeLabel>(?:\d+x)?(?:\d+\.)?\d+[A-Za-z](?:-[A-Za-z]+(\d+\.)?\d+[A-Za-z]+)?)(?:-(?<FineTune>[A-Za-z0-9\s-]+))?)?-(?:(?<Version>v\d+(?:\.\d+)*))(?:-(?<Encoding>(?!LoRA|vocab)[\w_]+))?(?:-(?<Type>LoRA|vocab))?(?:-(?<Shard>\d{5}-of-\d{5}))?\.gguf$` which will check that you got the minimum BaseName, SizeLabel and Version present in the correct order.

For example:

  * `Mixtral-8x7B-v0.1-KQ2.gguf`:
    - Model Name: Mixtral
    - Expert Count: 8
    - Parameter Count: 7B
    - Version Number: v0.1
    - Weight Encoding Scheme: KQ2

  * `Hermes-2-Pro-Llama-3-8B-F16.gguf`:
    - Model Name: Hermes 2 Pro Llama 3
    - Expert Count: 0
    - Parameter Count: 8B
    - Version Number: v1.0
    - Weight Encoding Scheme: F16
    - Shard: N/A

  * `Grok-100B-v1.0-Q4_0-00003-of-00009.gguf`
    - Model Name: Grok
    - Expert Count: 0
    - Parameter Count: 100B
    - Version Number: v1.0
    - Weight Encoding Scheme: Q4_0
    - Shard: 3 out of 9 total shards


<details><summary>Example Node.js Regex Function</summary>

```js
#!/usr/bin/env node
const ggufRegex = /^(?<BaseName>[A-Za-z0-9\s]*(?:(?:-(?:(?:[A-Za-z\s][A-Za-z0-9\s]*)|(?:[0-9\s]*)))*))-(?:(?<SizeLabel>(?:\d+x)?(?:\d+\.)?\d+[A-Za-z](?:-[A-Za-z]+(\d+\.)?\d+[A-Za-z]+)?)(?:-(?<FineTune>[A-Za-z0-9\s-]+))?)?-(?:(?<Version>v\d+(?:\.\d+)*))(?:-(?<Encoding>(?!LoRA|vocab)[\w_]+))?(?:-(?<Type>LoRA|vocab))?(?:-(?<Shard>\d{5}-of-\d{5}))?\.gguf$/;

function parseGGUFFilename(filename) {
  const match = ggufRegex.exec(filename);
  if (!match)
    return null;
  const {BaseName = null, SizeLabel = null, FineTune = null, Version = "v1.0", Encoding = null, Type = null, Shard = null} = match.groups;
  return {BaseName: BaseName, SizeLabel: SizeLabel, FineTune: FineTune, Version: Version, Encoding: Encoding, Type: Type, Shard: Shard};
}

const testCases = [
  {filename: 'Mixtral-8x7B-v0.1-KQ2.gguf',                         expected: { BaseName: 'Mixtral',              SizeLabel: '8x7B',     FineTune: null, Version: 'v0.1',   Encoding: 'KQ2',  Type: null, Shard: null}},
  {filename: 'Grok-100B-v1.0-Q4_0-00003-of-00009.gguf',            expected: { BaseName: 'Grok',                 SizeLabel: '100B',     FineTune: null, Version: 'v1.0',   Encoding: 'Q4_0', Type: null, Shard: "00003-of-00009"}},
  {filename: 'Hermes-2-Pro-Llama-3-8B-v1.0-F16.gguf',              expected: { BaseName: 'Hermes-2-Pro-Llama-3', SizeLabel: '8B', FineTune: null, Version: 'v1.0',   Encoding: 'F16',  Type: null, Shard: null}},
  {filename: 'Phi-3-mini-3.8B-ContextLength4k-instruct-v1.0.gguf', expected: { BaseName: 'Phi-3-mini',   SizeLabel: '3.8B-ContextLength4k', FineTune: 'instruct', Version: 'v1.0',   Encoding: null,  Type: null, Shard: null}},
  {filename: 'not-a-known-arrangement.gguf',                       expected: null},
];

testCases.forEach(({ filename, expected }) => {
  const result = parseGGUFFilename(filename);
  const passed = JSON.stringify(result) === JSON.stringify(expected);
  console.log(`${filename}: ${passed ? "PASS" : "FAIL"}`);
  if (!passed) {
      console.log(result);
      console.log(expected);
  }
});
```

</details>


### File Structure

![image](https://github.com/ggerganov/ggml/assets/1991296/c3623641-3a1d-408e-bfaf-1b7c4e16aa63)
*diagram by [@mishig25](https://github.com/mishig25) (GGUF v3)*

GGUF files are structured as follows. They use a global alignment specified in the `general.alignment` metadata field, referred to as `ALIGNMENT` below. Where required, the file is padded with `0x00` bytes to the next multiple of `general.alignment`.

Fields, including arrays, are written sequentially without alignment unless otherwise specified.

Models are little-endian by default. They can also come in big-endian for use with big-endian computers; in this case, all values (including metadata values and tensors) will also be big-endian. At the time of writing, there is no way to determine if a model is big-endian; this may be rectified in future versions. If no additional information is provided, assume the model is little-endian.

```c
enum ggml_type: uint32_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_COUNT,
};

enum gguf_metadata_value_type: uint32_t {
    // The value is a 8-bit unsigned integer.
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    // The value is a 8-bit signed integer.
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    // The value is a 16-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    // The value is a 32-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    // The value is a 64-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    // The value is a 64-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

// A string in GGUF.
struct gguf_string_t {
    // The length of the string, in bytes.
    uint64_t len;
    // The string as a UTF-8 non-null-terminated string.
    char string[len];
};

union gguf_metadata_value_t {
    uint8_t uint8;
    int8_t int8;
    uint16_t uint16;
    int16_t int16;
    uint32_t uint32;
    int32_t int32;
    float float32;
    uint64_t uint64;
    int64_t int64;
    double float64;
    bool bool_;
    gguf_string_t string;
    struct {
        // Any value type is valid, including arrays.
        gguf_metadata_value_type type;
        // Number of elements, not bytes
        uint64_t len;
        // The array of values.
        gguf_metadata_value_t array[len];
    } array;
};

struct gguf_metadata_kv_t {
    // The key of the metadata. It is a standard GGUF string, with the following caveats:
    // - It must be a valid ASCII string.
    // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
    // - It must be at most 2^16-1/65535 bytes long.
    // Any keys that do not follow these rules are invalid.
    gguf_string_t key;

    // The type of the value.
    // Must be one of the `gguf_metadata_value_type` values.
    gguf_metadata_value_type value_type;
    // The value.
    gguf_metadata_value_t value;
};

struct gguf_header_t {
    // Magic number to announce that this is a GGUF file.
    // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
    // Your executor might do little-endian byte order, so it might be
    // check for 0x46554747 and letting the endianness cancel out.
    // Consider being *very* explicit about the byte order here.
    uint32_t magic;
    // The version of the format implemented.
    // Must be `3` for version described in this spec, which introduces big-endian support.
    //
    // This version should only be increased for structural changes to the format.
    // Changes that do not affect the structure of the file should instead update the metadata
    // to signify the change.
    uint32_t version;
    // The number of tensors in the file.
    // This is explicit, instead of being included in the metadata, to ensure it is always present
    // for loading the tensors.
    uint64_t tensor_count;
    // The number of metadata key-value pairs.
    uint64_t metadata_kv_count;
    // The metadata key-value pairs.
    gguf_metadata_kv_t metadata_kv[metadata_kv_count];
};

uint64_t align_offset(uint64_t offset) {
    return offset + (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT;
}

struct gguf_tensor_info_t {
    // The name of the tensor. It is a standard GGUF string, with the caveat that
    // it must be at most 64 bytes long.
    gguf_string_t name;
    // The number of dimensions in the tensor.
    // Currently at most 4, but this may change in the future.
    uint32_t n_dimensions;
    // The dimensions of the tensor.
    uint64_t dimensions[n_dimensions];
    // The type of the tensor.
    ggml_type type;
    // The offset of the tensor's data in this file in bytes.
    //
    // This offset is relative to `tensor_data`, not to the start
    // of the file, to make it easier for writers to write the file.
    // Readers should consider exposing this offset relative to the
    // file to make it easier to read the data.
    //
    // Must be a multiple of `ALIGNMENT`. That is, `align_offset(offset) == offset`.
    uint64_t offset;
};

struct gguf_file_t {
    // The header of the file.
    gguf_header_t header;

    // Tensor infos, which can be used to locate the tensor data.
    gguf_tensor_info_t tensor_infos[header.tensor_count];

    // Padding to the nearest multiple of `ALIGNMENT`.
    //
    // That is, if `sizeof(header) + sizeof(tensor_infos)` is not a multiple of `ALIGNMENT`,
    // this padding is added to make it so.
    //
    // This can be calculated as `align_offset(position) - position`, where `position` is
    // the position of the end of `tensor_infos` (i.e. `sizeof(header) + sizeof(tensor_infos)`).
    uint8_t _padding[];

    // Tensor data.
    //
    // This is arbitrary binary data corresponding to the weights of the model. This data should be close
    // or identical to the data in the original model file, but may be different due to quantization or
    // other optimizations for inference. Any such deviations should be recorded in the metadata or as
    // part of the architecture definition.
    //
    // Each tensor's data must be stored within this array, and located through its `tensor_infos` entry.
    // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between tensors
    // should be padded to `ALIGNMENT` bytes.
    uint8_t tensor_data[];
};
```

## Standardized key-value pairs

The following key-value pairs are standardized. This list may grow in the future as more use cases are discovered. Where possible, names are shared with the original model definitions to make it easier to map between the two.

Not all of these are required, but they are all recommended. Keys that are required are bolded. For omitted pairs, the reader should assume that the value is unknown and either default or error as appropriate.

The community can develop their own key-value pairs to carry additional data. However, these should be namespaced with the relevant community name to avoid collisions. For example, the `rustformers` community might use `rustformers.` as a prefix for all of their keys.

If a particular community key is widely used, it may be promoted to a standardized key.

By convention, most counts/lengths/etc are `uint64` unless otherwise specified. This is to allow for larger models to be supported in the future. Some models may use `uint32` for their values; it is recommended that readers support both.

### General

#### Required

- **`general.architecture: string`**: describes what architecture this model implements. All lowercase ASCII, with only `[a-z0-9]+` characters allowed. Known values include:
  - `llama`
  - `mpt`
  - `gptneox`
  - `gptj`
  - `gpt2`
  - `bloom`
  - `falcon`
  - `mamba`
  - `rwkv`
- **`general.quantization_version: uint32`**: The version of the quantization format. Not required if the model is not quantized (i.e. no tensors are quantized). If any tensors are quantized, this _must_ be present. This is separate to the quantization scheme of the tensors itself; the quantization version may change without changing the scheme's name (e.g. the quantization scheme is Q5_K, and the quantization version is 4).
- **`general.alignment: uint32`**: the global alignment to use, as described above. This can vary to allow for different alignment schemes, but it must be a multiple of 8. Some writers may not write the alignment. If the alignment is **not** specified, assume it is `32`.

#### General metadata

- `general.name: string`: The name of the model. This should be a human-readable name that can be used to identify the model. It should be unique within the community that the model is defined in.
- `general.author: string`: The author of the model.
- `general.version: string`: The version of the model.
- `general.organization: string`: The organization of the model.
- `general.basename: string`: The base model name / architecture of the model
- `general.finetune: string`: What has the base model been optimized toward.
- `general.description: string`: free-form description of the model including anything that isn't covered by the other fields
- `general.quantized_by: string`: The name of the individual who quantized the model
- `general.size_label: string`: Size class of the model, such as number of weights and experts. (Useful for leader boards)
- `general.license: string`: License of the model, expressed as a [SPDX license expression](https://spdx.github.io/spdx-spec/v2-draft/SPDX-license-expressions/) (e.g. `"MIT OR Apache-2.0`). Do not include any other information, such as the license text or the URL to the license.
- `general.license.name: string`: Human friendly license name
- `general.license.link: string`: URL to the license.
- `general.url: string`: URL to the model's homepage. This can be a GitHub repo, a paper, etc.
- `general.doi: string`: Digital Object Identifier (DOI) https://www.doi.org/
- `general.uuid: string`: [Universally unique identifier](https://en.wikipedia.org/wiki/Universally_unique_identifier)
- `general.repo_url: string`: URL to the model's repository such as a GitHub repo or HuggingFace repo
- `general.tags: string[]`: List of tags that can be used as search terms for a search engine or social media
- `general.languages: string[]`: What languages can the model speak. Encoded as [ISO 639](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) two letter codes
- `general.datasets: string[]`: Links or references to datasets that the model was trained upon
- `general.file_type: uint32`: An enumerated value describing the type of the majority of the tensors in the file. Optional; can be inferred from the tensor types.
  - `ALL_F32 = 0`
  - `MOSTLY_F16 = 1`
  - `MOSTLY_Q4_0 = 2`
  - `MOSTLY_Q4_1 = 3`
  - `MOSTLY_Q4_1_SOME_F16 = 4`
  - `MOSTLY_Q4_2 = 5` (support removed)
  - `MOSTLY_Q4_3 = 6` (support removed)
  - `MOSTLY_Q8_0 = 7`
  - `MOSTLY_Q5_0 = 8`
  - `MOSTLY_Q5_1 = 9`
  - `MOSTLY_Q2_K = 10`
  - `MOSTLY_Q3_K_S = 11`
  - `MOSTLY_Q3_K_M = 12`
  - `MOSTLY_Q3_K_L = 13`
  - `MOSTLY_Q4_K_S = 14`
  - `MOSTLY_Q4_K_M = 15`
  - `MOSTLY_Q5_K_S = 16`
  - `MOSTLY_Q5_K_M = 17`
  - `MOSTLY_Q6_K = 18`

#### Source metadata

Information about where this model came from. This is useful for tracking the provenance of the model, and for finding the original source if the model is modified. For a model that was converted from GGML, for example, these keys would point to the model that was converted from.

- `general.source.url: string`: URL to the source of the model's homepage. This can be a GitHub repo, a paper, etc.
- `general.source.doi: string`: Source Digital Object Identifier (DOI) https://www.doi.org/
- `general.source.uuid: string`: Source [Universally unique identifier](https://en.wikipedia.org/wiki/Universally_unique_identifier)
- `general.source.repo_url: string`: URL to the source of the model's repository such as a GitHub repo or HuggingFace repo

- `general.base_model.count: uint32`: Number of parent models
- `general.base_model.{id}.name: string`: The name of the parent model.
- `general.base_model.{id}.author: string`: The author of the parent model.
- `general.base_model.{id}.version: string`: The version of the parent model.
- `general.base_model.{id}.organization: string`: The organization of the parent model.
- `general.base_model.{id}.url: string`: URL to the source of the parent model's homepage. This can be a GitHub repo, a paper, etc.
- `general.base_model.{id}.doi: string`: Parent Digital Object Identifier (DOI) https://www.doi.org/
- `general.base_model.{id}.uuid: string`: Parent [Universally unique identifier](https://en.wikipedia.org/wiki/Universally_unique_identifier)
- `general.base_model.{id}.repo_url: string`: URL to the source of the parent model's repository such as a GitHub repo or HuggingFace repo

### LLM

In the following, `[llm]` is used to fill in for the name of a specific LLM architecture. For example, `llama` for LLaMA, `mpt` for MPT, etc. If mentioned in an architecture's section, it is required for that architecture, but not all keys are required for all architectures. Consult the relevant section for more information.

- `[llm].context_length: uint64`: Also known as `n_ctx`. length of the context (in tokens) that the model was trained on. For most architectures, this is the hard limit on the length of the input. Architectures, like RWKV, that are not reliant on transformer-style attention may be able to handle larger inputs, but this is not guaranteed.
- `[llm].embedding_length: uint64`: Also known as `n_embd`. Embedding layer size.
- `[llm].block_count: uint64`: The number of blocks of attention+feed-forward layers (i.e. the bulk of the LLM). Does not include the input or embedding layers.
- `[llm].feed_forward_length: uint64`: Also known as `n_ff`. The length of the feed-forward layer.
- `[llm].use_parallel_residual: bool`: Whether or not the parallel residual logic should be used.
- `[llm].tensor_data_layout: string`: When a model is converted to GGUF, tensors may be rearranged to improve performance. This key describes the layout of the tensor data. This is not required; if not present, it is assumed to be `reference`.
  - `reference`: tensors are laid out in the same order as the original model
  - further options can be found for each architecture in their respective sections
- `[llm].expert_count: uint32`: Number of experts in MoE models (optional for non-MoE arches).
- `[llm].expert_used_count: uint32`: Number of experts used during each token token evaluation (optional for non-MoE arches).

#### Attention

- `[llm].attention.head_count: uint64`: Also known as `n_head`. Number of attention heads.
- `[llm].attention.head_count_kv: uint64`: The number of heads per group used in Grouped-Query-Attention. If not present or if present and equal to `[llm].attention.head_count`, the model does not use GQA.
- `[llm].attention.max_alibi_bias: float32`: The maximum bias to use for ALiBI.
- `[llm].attention.clamp_kqv: float32`: Value (`C`) to clamp the values of the `Q`, `K`, and `V` tensors between (`[-C, C]`).
- `[llm].attention.layer_norm_epsilon: float32`: Layer normalization epsilon.
- `[llm].attention.layer_norm_rms_epsilon: float32`: Layer RMS normalization epsilon.
- `[llm].attention.key_length: uint32`: The optional size of a key head, $d_k$. If not specified, it will be `n_embd / n_head`.
- `[llm].attention.value_length: uint32`: The optional size of a value head, $d_v$. If not specified, it will be `n_embd / n_head`.

#### RoPE

- `[llm].rope.dimension_count: uint64`: The number of rotary dimensions for RoPE.
- `[llm].rope.freq_base: float32`: The base frequency for RoPE.

##### Scaling

The following keys describe RoPE scaling parameters:

- `[llm].rope.scaling.type: string`: Can be `none`, `linear`, or `yarn`.
- `[llm].rope.scaling.factor: float32`: A scale factor for RoPE to adjust the context length.
- `[llm].rope.scaling.original_context_length: uint32_t`: The original context length of the base model.
- `[llm].rope.scaling.finetuned: bool`: True if model has been finetuned with RoPE scaling.

Note that older models may not have these keys, and may instead use the following key:

- `[llm].rope.scale_linear: float32`: A linear scale factor for RoPE to adjust the context length.

It is recommended that models use the newer keys if possible, as they are more flexible and allow for more complex scaling schemes. Executors will need to support both indefinitely.

#### SSM

- `[llm].ssm.conv_kernel: uint32`: The size of the rolling/shift state.
- `[llm].ssm.inner_size: uint32`: The embedding size of the states.
- `[llm].ssm.state_size: uint32`: The size of the recurrent state.
- `[llm].ssm.time_step_rank: uint32`: The rank of time steps.

#### Models

The following sections describe the metadata for each model architecture. Each key specified _must_ be present.

##### LLaMA

- `llama.context_length`
- `llama.embedding_length`
- `llama.block_count`
- `llama.feed_forward_length`
- `llama.rope.dimension_count`
- `llama.attention.head_count`
- `llama.attention.layer_norm_rms_epsilon`

###### Optional

- `llama.rope.scale`
- `llama.attention.head_count_kv`
- `llama.tensor_data_layout`:
  - `Meta AI original pth`:
    ```python
    def permute(weights: NDArray, n_head: int) -> NDArray:
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                    .swapaxes(1, 2)
                    .reshape(weights.shape))
    ```
- `llama.expert_count`
- `llama.expert_used_count`

##### MPT

- `mpt.context_length`
- `mpt.embedding_length`
- `mpt.block_count`
- `mpt.attention.head_count`
- `mpt.attention.alibi_bias_max`
- `mpt.attention.clip_kqv`
- `mpt.attention.layer_norm_epsilon`

##### GPT-NeoX

- `gptneox.context_length`
- `gptneox.embedding_length`
- `gptneox.block_count`
- `gptneox.use_parallel_residual`
- `gptneox.rope.dimension_count`
- `gptneox.attention.head_count`
- `gptneox.attention.layer_norm_epsilon`

###### Optional

- `gptneox.rope.scale`

##### GPT-J

- `gptj.context_length`
- `gptj.embedding_length`
- `gptj.block_count`
- `gptj.rope.dimension_count`
- `gptj.attention.head_count`
- `gptj.attention.layer_norm_epsilon`

###### Optional

- `gptj.rope.scale`

##### GPT-2

- `gpt2.context_length`
- `gpt2.embedding_length`
- `gpt2.block_count`
- `gpt2.attention.head_count`
- `gpt2.attention.layer_norm_epsilon`

##### BLOOM

- `bloom.context_length`
- `bloom.embedding_length`
- `bloom.block_count`
- `bloom.feed_forward_length`
- `bloom.attention.head_count`
- `bloom.attention.layer_norm_epsilon`

##### Falcon

- `falcon.context_length`
- `falcon.embedding_length`
- `falcon.block_count`
- `falcon.attention.head_count`
- `falcon.attention.head_count_kv`
- `falcon.attention.use_norm`
- `falcon.attention.layer_norm_epsilon`

###### Optional

- `falcon.tensor_data_layout`:

  - `jploski` (author of the original GGML implementation of Falcon):

    ```python
    # The original query_key_value tensor contains n_head_kv "kv groups",
    # each consisting of n_head/n_head_kv query weights followed by one key
    # and one value weight (shared by all query heads in the kv group).
    # This layout makes it a big pain to work with in GGML.
    # So we rearrange them here,, so that we have n_head query weights
    # followed by n_head_kv key weights followed by n_head_kv value weights,
    # in contiguous fashion.

    if "query_key_value" in src:
        qkv = model[src].view(
            n_head_kv, n_head // n_head_kv + 2, head_dim, head_dim * n_head)

        q = qkv[:, :-2 ].reshape(n_head * head_dim, head_dim * n_head)
        k = qkv[:, [-2]].reshape(n_head_kv * head_dim, head_dim * n_head)
        v = qkv[:, [-1]].reshape(n_head_kv * head_dim, head_dim * n_head)

        model[src] = torch.cat((q,k,v)).reshape_as(model[src])
    ```

##### Mamba

- `mamba.context_length`
- `mamba.embedding_length`
- `mamba.block_count`
- `mamba.ssm.conv_kernel`
- `mamba.ssm.inner_size`
- `mamba.ssm.state_size`
- `mamba.ssm.time_step_rank`
- `mamba.attention.layer_norm_rms_epsilon`

##### RWKV

The vocabulary size is the same as the number of rows in the `head` matrix.

- `rwkv.architecture_version: uint32`: The only allowed value currently is 4. Version 5 is expected to appear some time in the future.
- `rwkv.context_length: uint64`: Length of the context used during training or fine-tuning. RWKV is able to handle larger context than this limit, but the output quality may suffer.
- `rwkv.block_count: uint64`
- `rwkv.embedding_length: uint64`
- `rwkv.feed_forward_length: uint64`

##### Whisper

Keys that do not have types defined should be assumed to share definitions with `llm.` keys.
(For example, `whisper.context_length` is equivalent to `llm.context_length`.)
This is because they are both transformer models.

- `whisper.encoder.context_length`
- `whisper.encoder.embedding_length`
- `whisper.encoder.block_count`
- `whisper.encoder.mels_count: uint64`
- `whisper.encoder.attention.head_count`

- `whisper.decoder.context_length`
- `whisper.decoder.embedding_length`
- `whisper.decoder.block_count`
- `whisper.decoder.attention.head_count`

#### Prompting

**TODO**: Include prompt format, and/or metadata about how it should be used (instruction, conversation, autocomplete, etc).

### LoRA

**TODO**: Figure out what metadata is needed for LoRA. Probably desired features:

- match an existing model exactly, so that it can't be misapplied
- be marked as a LoRA so executors won't try to run it by itself

Should this be an architecture, or should it share the details of the original model with additional fields to mark it as a LoRA?

### Tokenizer

The following keys are used to describe the tokenizer of the model. It is recommended that model authors support as many of these as possible, as it will allow for better tokenization quality with supported executors.

#### GGML

GGML supports an embedded vocabulary that enables inference of the model, but implementations of tokenization using this vocabulary (i.e. `llama.cpp`'s tokenizer) may have lower accuracy than the original tokenizer used for the model. When a more accurate tokenizer is available and supported, it should be used instead.

It is not guaranteed to be standardized across models, and may change in the future. It is recommended that model authors use a more standardized tokenizer if possible.

- `tokenizer.ggml.model: string`: The name of the tokenizer model.
  - `llama`: Llama style SentencePiece (tokens and scores extracted from HF `tokenizer.model`)
  - `replit`: Replit style SentencePiece (tokens and scores extracted from HF `spiece.model`)
  - `gpt2`: GPT-2 / GPT-NeoX style BPE (tokens extracted from HF `tokenizer.json`)
  - `rwkv`: RWKV tokenizer
- `tokenizer.ggml.tokens: array[string]`: A list of tokens indexed by the token ID used by the model.
- `tokenizer.ggml.scores: array[float32]`: If present, the score/probability of each token. If not present, all tokens are assumed to have equal probability. If present, it must have the same length and index as `tokens`.
- `tokenizer.ggml.token_type: array[int32]`: The token type (1=normal, 2=unknown, 3=control, 4=user defined, 5=unused, 6=byte). If present, it must have the same length and index as `tokens`.
- `tokenizer.ggml.merges: array[string]`: If present, the merges of the tokenizer. If not present, the tokens are assumed to be atomic.
- `tokenizer.ggml.added_tokens: array[string]`: If present, tokens that were added after training.

##### Special tokens

- `tokenizer.ggml.bos_token_id: uint32`: Beginning of sequence marker
- `tokenizer.ggml.eos_token_id: uint32`: End of sequence marker
- `tokenizer.ggml.unknown_token_id: uint32`: Unknown token
- `tokenizer.ggml.separator_token_id: uint32`: Separator token
- `tokenizer.ggml.padding_token_id: uint32`: Padding token

#### Hugging Face

Hugging Face maintains their own `tokenizers` library that supports a wide variety of tokenizers. If your executor uses this library, it may be able to use the model's tokenizer directly.

- `tokenizer.huggingface.json: string`: the entirety of the HF `tokenizer.json` for a given model (e.g. <https://huggingface.co/mosaicml/mpt-7b-instruct/blob/main/tokenizer.json>). Included for compatibility with executors that support HF tokenizers directly.

#### Other

Other tokenizers may be used, but are not necessarily standardized. They may be executor-specific. They will be documented here as they are discovered/further developed.

- `tokenizer.rwkv.world: string`: a RWKV World tokenizer, like [this](https://github.com/BlinkDL/ChatRWKV/blob/main/tokenizer/rwkv_vocab_v20230424.txt). This text file should be included verbatim.
- `tokenizer.chat_template : string`: a Jinja template that specifies the input format expected by the model. For more details see: <https://huggingface.co/docs/transformers/main/en/chat_templating>

### Computation graph

This is a future extension and still needs to be discussed, and may necessitate a new GGUF version. At the time of writing, the primary blocker is the stabilization of the computation graph format.

A sample computation graph of GGML nodes could be included in the model itself, allowing an executor to run the model without providing its own implementation of the architecture. This would allow for a more consistent experience across executors, and would allow for more complex architectures to be supported without requiring the executor to implement them.

## Standardized tensor names

To minimize complexity and maximize compatibility, it is recommended that models using the transformer architecture use the following naming convention for their tensors:

### Base layers

`AA.weight` `AA.bias`

where `AA` can be:

- `token_embd`: Token embedding layer
- `pos_embd`: Position embedding layer
- `output_norm`: Output normalization layer
- `output`: Output layer

### Attention and feed-forward layer blocks

`blk.N.BB.weight` `blk.N.BB.bias`

where N signifies the block number a layer belongs to, and where `BB` could be:

- `attn_norm`: Attention normalization layer
- `attn_norm_2`: Attention normalization layer
- `attn_qkv`: Attention query-key-value layer
- `attn_q`: Attention query layer
- `attn_k`: Attention key layer
- `attn_v`: Attention value layer
- `attn_output`: Attention output layer

- `ffn_norm`: Feed-forward network normalization layer
- `ffn_up`: Feed-forward network "up" layer
- `ffn_gate`: Feed-forward network "gate" layer
- `ffn_down`: Feed-forward network "down" layer
- `ffn_gate_inp`: Expert-routing layer for the Feed-forward network in MoE models
- `ffn_gate_exp`: Feed-forward network "gate" layer per expert in MoE models
- `ffn_down_exp`: Feed-forward network "down" layer per expert in MoE models
- `ffn_up_exp`: Feed-forward network "up" layer per expert in MoE models

- `ssm_in`: State space model input projections layer
- `ssm_conv1d`: State space model rolling/shift layer
- `ssm_x`: State space model selective parametrization layer
- `ssm_a`: State space model state compression layer
- `ssm_d`: State space model skip connection layer
- `ssm_dt`: State space model time step layer
- `ssm_out`: State space model output projection layer

## Version History

This document is actively updated to describe the current state of the metadata, and these changes are not tracked outside of the commits.

However, the format _itself_ has changed. The following sections describe the changes to the format itself.

### v3

Adds big-endian support.

### v2

Most countable values (lengths, etc) were changed from `uint32` to `uint64` to allow for larger models to be supported in the future.

### v1

Initial version.

## Historical State of Affairs

The following information is provided for context, but is not necessary to understand the rest of this document.

### Overview

At present, there are three GGML file formats floating around for LLMs:

- **GGML** (unversioned): baseline format, with no versioning or alignment.
- **GGMF** (versioned): the same as GGML, but with versioning. Only one version exists.
- **GGJT**: Aligns the tensors to allow for use with `mmap`, which requires alignment. v1, v2 and v3 are identical, but the latter versions use a different quantization scheme that is incompatible with previous versions.

GGML is primarily used by the examples in `ggml`, while GGJT is used by `llama.cpp` models. Other executors may use any of the three formats, but this is not 'officially' supported.

These formats share the same fundamental structure:

- a magic number with an optional version number
- model-specific hyperparameters, including
  - metadata about the model, such as the number of layers, the number of heads, etc.
  - a `ftype` that describes the type of the majority of the tensors,
    - for GGML files, the quantization version is encoded in the `ftype` divided by 1000
- an embedded vocabulary, which is a list of strings with length prepended. The GGMF/GGJT formats embed a float32 score next to the strings.
- finally, a list of tensors with their length-prepended name, type, and (aligned, in the case of GGJT) tensor data

Notably, this structure does not identify what model architecture the model belongs to, nor does it offer any flexibility for changing the structure of the hyperparameters. This means that the only way to add new hyperparameters is to add them to the end of the list, which is a breaking change for existing models.

### Drawbacks

Unfortunately, over the last few months, there are a few issues that have become apparent with the existing models:

- There's no way to identify which model architecture a given model is for, because that information isn't present
  - Similarly, existing programs cannot intelligently fail upon encountering new architectures
- Adding or removing any new hyperparameters is a breaking change, which is impossible for a reader to detect without using heuristics
- Each model architecture requires its own conversion script to their architecture's variant of GGML
- Maintaining backwards compatibility without breaking the structure of the format requires clever tricks, like packing the quantization version into the ftype, which are not guaranteed to be picked up by readers/writers, and are not consistent between the two formats

### Why not other formats?

There are a few other formats that could be used, but issues include:

- requiring additional dependencies to load or save the model, which is complicated in a C environment
- limited or no support for 4-bit quantization
- existing cultural expectations (e.g. whether or not the model is a directory or a file)
- lack of support for embedded vocabularies
- lack of control over direction of future development

Ultimately, it is likely that GGUF will remain necessary for the foreseeable future, and it is better to have a single format that is well-documented and supported by all executors than to contort an existing format to fit the needs of GGML.

*/

/*
version
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
    pub fn encode_text_to_gguf_tokens(&self, text: &str) -> io::Result<Vec<u32>> {
        let mut tokens = Vec::new();
        
        // Add BOS token
        tokens.push(self.bos_token);
        
        // TODO: Implement actual tokenization
        // This is placeholder - needs proper subword tokenization
        for word in text.split_whitespace() {
            if let Some(&token_id) = self.vocab.get(word) {
                tokens.push(token_id);
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("Token not in GGUF vocabulary: {}", word)
                ));
            }
        }
        
        Ok(tokens)
    }

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
    pub fn decode_gguf_tokens_to_text(&self, token_ids: &[u32]) -> io::Result<String> {
        let mut text = Vec::new();
        
        for &token_id in token_ids {
            // Skip special tokens
            if token_id == self.bos_token || token_id == self.eos_token {
                continue;
            }
            
            if let Some(token) = self.id_to_token.get(&token_id) {
                text.push(token.clone());
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Token ID not in GGUF vocabulary: {}", token_id)
                ));
            }
        }
        
        Ok(text.join(" "))  // Simplified joining - needs proper detokenization
    }
}

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
    
    Ok(TokenizerGGUF {
        vocab,
        id_to_token,
        bos_token,
        eos_token,
    })
}


*/

use std::fs::File;
use std::io::{
    self, 
    Read, 
    // Seek, 
    // SeekFrom
};
use byteorder::{
    LittleEndian, 
    ReadBytesExt
};
use std::collections::HashMap;
use memmap2::Mmap;

mod gguf_tokenizer_module;  // looks for either gguf_tokenizer_module_tokenizer.rs or gguf_tokenizer_module/mod.rs

const GGUF_MAGIC: u32 = 0x46554747;  // "GGUF" in ASCII

/// Represents the core architecture parameters of a model
/// 
/// # Fields
/// * `model_type` - Architecture type (llama, gpt2, etc)
/// * `hidden_size` - Main embedding dimension
/// * `num_attention_heads` - Number of attention heads
/// * `num_kv_heads` - Number of key/value heads (for GQA, equals num_attention_heads if not GQA)
/// * `num_layers` - Number of transformer blocks
/// * `vocab_size` - Size of vocabulary (output dimension)
/// * `intermediate_size` - FFN hidden dimension
#[derive(Debug)]
struct ModelArchitecture {
    model_type: String,
    hidden_size: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    num_layers: usize,
    vocab_size: usize,
    intermediate_size: usize,
}

impl GGUFModel {
    
    /// Loads tokenizer for this model
    /// 
    /// # Returns
    /// * `io::Result<Tokenizer>` - Tokenizer for this model
    /// 
    /// # Errors
    /// * `io::Error` if required tokenizer data not found
    /// Loads GGUF-specific tokenizer for this model
    pub fn load_tokenizer(&self) -> io::Result<gguf_tokenizer_module::TokenizerGGUF> {
        gguf_tokenizer_module::load_from_gguf(self)
    }
    
    /// High-level inference function that takes a string and returns a string
    /// 
    /// # Arguments
    /// * `input_text` - The text to process
    /// 
    /// # Returns
    /// * `Result<String>` - The model's output text
    pub fn infer_string(&self, input_text: &str) -> io::Result<String> {
        // 1. Load tokenizer
        let tokenizer = self.load_tokenizer()?;
        
        // 2. Encode input text to tokens
        println!("Encoding input text: {}", input_text);
        let tokens = tokenizer.encode_text_to_gguf_tokens(input_text)?;
        println!("Encoded to tokens: {:?}", tokens);
        
        // 3. Run inference on tokens
        let mut output_logits = Vec::new();
        for &token_id in &tokens {
            let logits = self.run_model_inference(token_id as usize)?;
            output_logits.push(logits);
        }
        
        // 4. Get most likely token for each logit vector
        let mut output_tokens = Vec::new();
        for logits in &output_logits {
            // Find index of maximum logit
            let mut max_idx = 0;
            let mut max_val = logits[0];
            for (i, &val) in logits.iter().enumerate() {
                if val > max_val {
                    max_idx = i;
                    max_val = val;
                }
            }
            output_tokens.push(max_idx as u32);
        }
        
        // 5. Decode tokens back to text
        let output_text = tokenizer.decode_gguf_tokens_to_text(&output_tokens)?;
        
        Ok(output_text)
    }
    
    
    /// Inspects model metadata to determine architecture parameters
    /// 
    /// # Returns
    /// * `ModelArchitecture` containing all parameters needed for inference
    /// 
    /// # Errors
    /// * `io::Error` if required metadata is missing or invalid
    /// * `io::Error` if model type is unsupported
    /// 
    /// # Example
    /// ```
    /// let model = GGUFModel::load("model.gguf")?;
    /// let arch = model.inspect_architecture()?;
    /// println!("Model has {} layers and {} attention heads", 
    ///          arch.num_layers, arch.num_attention_heads);
    /// ```
    fn inspect_architecture(&self) -> io::Result<ModelArchitecture> {
        // Get model type from metadata
        let model_type = self.header.metadata.get_string("general.architecture")
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Model architecture not found"))?
            .to_string();

        // Get architecture parameters based on model type
        let (hidden_size, num_heads, num_layers) = match model_type.as_ref() {
            "llama" => {
                let hidden = self.get_metadata_dim("llama.embedding_length")?;
                let heads = self.get_metadata_dim("llama.attention.head_count")?;
                let layers = self.get_metadata_dim("llama.block_count")?;
                (hidden, heads, layers)
            },
            "gpt2" => {
                let hidden = self.get_metadata_dim("gpt2.embedding_length")?;
                let heads = self.get_metadata_dim("gpt2.attention.head_count")?;
                let layers = self.get_metadata_dim("gpt2.block_count")?;
                (hidden, heads, layers)
            },
            // Add other model types here
            _ => return Err(io::Error::new(io::ErrorKind::Unsupported, 
                          format!("Unsupported model type: {}", model_type)))
        };

        // Handle Grouped-Query Attention (GQA)
        let num_kv_heads = match model_type.as_ref() {
            "llama" => self.get_metadata_dim("llama.attention.head_count_kv")
                          .unwrap_or(num_heads),
            _ => num_heads  // Default to regular attention if not specified
        };

        let arch = ModelArchitecture {
            model_type: model_type.clone(),
            hidden_size,
            num_attention_heads: num_heads,
            num_kv_heads,
            num_layers,
            vocab_size: self.get_vocab_size()?,
            intermediate_size: self.get_intermediate_size(&model_type)?,
        };

        // Print architecture information for debugging
        println!("\nModel Architecture Details:");
        println!("- Type: {}", arch.model_type);
        println!("- Hidden size: {}", arch.hidden_size);
        println!("- Attention heads: {}", arch.num_attention_heads);
        println!("- KV heads: {}", arch.num_kv_heads);
        println!("- Layers: {}", arch.num_layers);
        println!("- Vocabulary size: {}", arch.vocab_size);
        println!("- Intermediate size: {}", arch.intermediate_size);
        
        Ok(arch)
    }

    /// Helper function to extract dimension values from metadata
    /// 
    /// # Arguments
    /// * `key` - Metadata key to look up
    /// 
    /// # Returns
    /// * `usize` - The dimension value
    /// 
    /// # Errors
    /// * `io::Error` if key not found or value is invalid type
    fn get_metadata_dim(&self, key: &str) -> io::Result<usize> {
        match &self.header.metadata.kvs.get(key) {
            Some(MetadataValue::U64(v)) => Ok(*v as usize),
            Some(MetadataValue::U32(v)) => Ok(*v as usize),
            _ => Err(io::Error::new(io::ErrorKind::NotFound, 
                     format!("Dimension not found: {}", key)))
        }
    }

    /// Determines vocabulary size from output tensor dimensions
    /// 
    /// # Returns
    /// * `usize` - Size of model vocabulary
    /// 
    /// # Errors
    /// * `io::Error` if output tensor not found
    fn get_vocab_size(&self) -> io::Result<usize> {
        let output_tensor = self.tensors.iter()
            .find(|t| t.name == "output.weight")
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Output tensor not found"))?;
        
        Ok(output_tensor.dimensions[1] as usize)
    }

    /// Gets intermediate (FFN) size based on model type
    /// 
    /// # Arguments
    /// * `model_type` - Type of model architecture
    /// 
    /// # Returns
    /// * `usize` - Size of FFN intermediate layer
    /// 
    /// # Errors
    /// * `io::Error` if size not found or model type unsupported
    fn get_intermediate_size(&self, model_type: &str) -> io::Result<usize> {
        match model_type {
            "llama" => self.get_metadata_dim("llama.feed_forward_length"),
            // Add other model types here
            _ => Err(io::Error::new(io::ErrorKind::Unsupported, 
                     format!("Unsupported model type for FFN: {}", model_type)))
        }
    }
}



#[derive(Debug, Clone)]
struct GGUFString {
    data: String,
}

#[derive(Debug, Clone)]
enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(GGUFString),
    Array(Vec<MetadataValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Debug)]
struct ModelMetadata {
    kvs: HashMap<String, MetadataValue>,
}

impl ModelMetadata {
    fn get_string(&self, key: &str) -> Option<&str> {
        match self.kvs.get(key) {
            Some(MetadataValue::String(s)) => Some(&s.data),
            _ => None,
        }
    }

    fn get_u32(&self, key: &str) -> Option<u32> {
        match self.kvs.get(key) {
            Some(MetadataValue::U32(v)) => Some(*v),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct ModelHeader {
    version: u32,
    tensor_count: u64,
    metadata: ModelMetadata,
}

#[derive(Debug, Clone, Copy)]
enum GGMLType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
}

impl GGMLType {
    fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(GGMLType::F32),
            1 => Some(GGMLType::F16),
            2 => Some(GGMLType::Q4_0),
            3 => Some(GGMLType::Q4_1),
            6 => Some(GGMLType::Q5_0),
            7 => Some(GGMLType::Q5_1),
            8 => Some(GGMLType::Q8_0),
            9 => Some(GGMLType::Q8_1),
            _ => None,
        }
    }

    fn block_size(&self) -> usize {
        match self {
            GGMLType::F32 => 4,
            GGMLType::F16 => 2,
            GGMLType::Q4_0 | GGMLType::Q4_1 => 1,
            GGMLType::Q5_0 | GGMLType::Q5_1 => 1,
            GGMLType::Q8_0 | GGMLType::Q8_1 => 1,
        }
    }
}

#[derive(Debug)]
struct TensorInfo {
    name: String,
    dimensions: Vec<u64>,
    tensor_type: GGMLType,
    offset: u64,
}

impl TensorInfo {
    fn read_from(file: &mut File) -> io::Result<Self> {
        let name_len = file.read_u64::<LittleEndian>()?;
        let mut name_bytes = vec![0u8; name_len as usize];
        file.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let n_dims = file.read_u32::<LittleEndian>()?;
        let mut dimensions = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dimensions.push(file.read_u64::<LittleEndian>()?);
        }

        let type_val = file.read_u32::<LittleEndian>()?;
        let tensor_type = GGMLType::from_u32(type_val)
            .ok_or_else(|| io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid tensor type: {}", type_val)
            ))?;
        
        let offset = file.read_u64::<LittleEndian>()?;

        Ok(TensorInfo {
            name,
            dimensions,
            tensor_type,
            offset,
        })
    }

    fn element_count(&self) -> u64 {
        self.dimensions.iter().product()
    }

    fn byte_size(&self) -> u64 {
        self.element_count() * self.tensor_type.block_size() as u64
    }
}


fn read_metadata_value(file: &mut File, value_type: u32) -> io::Result<MetadataValue> {
    Ok(match value_type {
        0 => MetadataValue::U8(file.read_u8()?),
        1 => MetadataValue::I8(file.read_i8()?),
        2 => MetadataValue::U16(file.read_u16::<LittleEndian>()?),
        3 => MetadataValue::I16(file.read_i16::<LittleEndian>()?),
        4 => MetadataValue::U32(file.read_u32::<LittleEndian>()?),
        5 => MetadataValue::I32(file.read_i32::<LittleEndian>()?),
        6 => MetadataValue::F32(file.read_f32::<LittleEndian>()?),
        7 => MetadataValue::Bool(file.read_u8()? != 0),
        8 => {
            let len = file.read_u64::<LittleEndian>()?;
            let mut buffer = vec![0u8; len as usize];
            file.read_exact(&mut buffer)?;
            let data = String::from_utf8(buffer)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            MetadataValue::String(GGUFString { data })
        },
        9 => {
            let array_type = file.read_u32::<LittleEndian>()?;
            let array_len = file.read_u64::<LittleEndian>()?;
            let mut values = Vec::with_capacity(array_len as usize);
            for _ in 0..array_len {
                values.push(read_metadata_value(file, array_type)?);
            }
            MetadataValue::Array(values)
        },
        10 => MetadataValue::U64(file.read_u64::<LittleEndian>()?),
        11 => MetadataValue::I64(file.read_i64::<LittleEndian>()?),
        12 => MetadataValue::F64(file.read_f64::<LittleEndian>()?),
        _ => return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported metadata value type: {}", value_type)
        )),
    })
}

fn read_gguf_header_and_tensors(file: &mut File) -> io::Result<(ModelHeader, Vec<TensorInfo>)> {
    let magic = file.read_u32::<LittleEndian>()?;
    if magic != GGUF_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid GGUF magic number"
        ));
    }

    let version = file.read_u32::<LittleEndian>()?;
    if version != 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported GGUF version: {}", version)
        ));
    }

    let tensor_count = file.read_u64::<LittleEndian>()?;
    let metadata_count = file.read_u64::<LittleEndian>()?;

    let mut metadata_kvs = HashMap::new();
    for _ in 0..metadata_count {
        let key_len = file.read_u64::<LittleEndian>()?;
        let mut key_bytes = vec![0u8; key_len as usize];
        file.read_exact(&mut key_bytes)?;
        let key = String::from_utf8(key_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let value_type = file.read_u32::<LittleEndian>()?;
        let value = read_metadata_value(file, value_type)?;
        metadata_kvs.insert(key, value);
    }

    let header = ModelHeader {
        version,
        tensor_count,
        metadata: ModelMetadata { kvs: metadata_kvs },
    };

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        tensors.push(TensorInfo::read_from(file)?);
    }

    Ok((header, tensors))
}


struct GGUFModel {
    mmap: Mmap,
    header: ModelHeader,
    tensors: Vec<TensorInfo>,
}

impl GGUFModel {
    fn load(path: &str) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let (header, tensors) = read_gguf_header_and_tensors(&mut file)?;
        
        // Create memory map after we're done reading the header and tensor info
        let mmap = unsafe { Mmap::map(&file)? };
        
        Ok(GGUFModel {
            mmap,
            header,
            tensors,
        })
    }

    fn inspect_tokenizer(&self) -> io::Result<()> {
        println!("\nTokenizer Information:");
        
        // Get tokenizer type
        if let Some(MetadataValue::String(model_type)) = self.header.metadata.kvs.get("tokenizer.ggml.model") {
            println!("Tokenizer type: {}", model_type.data);
        }

        // Get vocabulary
        if let Some(MetadataValue::Array(tokens)) = self.header.metadata.kvs.get("tokenizer.ggml.tokens") {
            println!("Vocabulary size: {}", tokens.len());
            
            // Print first few tokens as example
            println!("\nFirst 16 tokens:");
            for (i, token) in tokens.iter().take(16).enumerate() {
                if let MetadataValue::String(s) = token {
                    println!("{}: '{}'", i, s.data);
                }
            }
        }

        // Get special token IDs
        let special_tokens = [
            "bos_token_id",
            "eos_token_id",
            "unknown_token_id",
            "separator_token_id",
            "padding_token_id"
        ];

        println!("\nSpecial tokens:");
        for token_name in special_tokens {
            let key = format!("tokenizer.ggml.{}", token_name);
            if let Some(MetadataValue::U32(id)) = self.header.metadata.kvs.get(&key) {
                println!("{}: {}", token_name, id);
            }
        }

        Ok(())
    }

    fn get_tensor_data(&self, tensor_idx: usize) -> io::Result<&[u8]> {
        let tensor = &self.tensors[tensor_idx];
        let start = tensor.offset as usize;
        let end = start + tensor.byte_size() as usize;
        
        if end > self.mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Tensor extends beyond file size"
            ));
        }
        
        Ok(&self.mmap[start..end])
    }
        
        
    // /*
    // The FFN implementation includes:

    // Layer normalization
    // Up/down projections
    // SwiGLU activation function
    // Memory-efficient sequential processing
    // Detailed logging for debugging
    // */
        
    /// Runs full model inference on a single token
    /// 
    /// # Design Notes
    /// Processing sequence:
    /// 1. Token embedding lookup
    /// 2. Process through all transformer blocks
    /// 3. Final layer norm
    /// 4. Project to vocabulary logits
    /// 
    /// Memory efficiency:
    /// - Processes one block at a time
    /// - Only loads needed weights through mmap
    /// - Maintains running hidden state
    fn run_model_inference(&self, token_id: usize) -> io::Result<Vec<f32>> {
        self.log_memory("Start model inference");
        
        // Get number of layers from metadata
        let n_blocks = match &self.header.metadata.kvs.get("llama.block_count") {
            Some(MetadataValue::U64(blocks)) => *blocks as usize,
            Some(MetadataValue::U32(blocks)) => *blocks as usize,
            _ => return Err(io::Error::new(io::ErrorKind::NotFound, "Block count not found"))
        };
        
        println!("Model architecture:");
        println!("- Number of blocks: {}", n_blocks);
        
        // 1. Get token embedding
        let mut hidden_states = self.get_token_embedding(token_id)?;
        println!("Initial embedding shape: {}", hidden_states.len());
        
        // 2. Process through transformer blocks
        for block_idx in 0..n_blocks {
            println!("\nProcessing block {}/{}", block_idx + 1, n_blocks);
            hidden_states = self.transformer_block(block_idx, &hidden_states)?;
            
            // Print sample values every few blocks
            if block_idx % 4 == 0 || block_idx == n_blocks - 1 {
                println!("Block {} output first value: {}", block_idx, hidden_states[0]);
            }
        }
        
        // 3. Final layer normalization
        let norm_idx = self.tensors.iter()
            .position(|t| t.name == "output_norm.weight")
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Output norm weights not found"))?;
            
        let normalized = self.apply_final_norm(norm_idx, &hidden_states)?;
        
        // 4. Project to vocabulary logits
        let logits = self.project_to_vocab(&normalized)?;
        
        self.log_memory("End model inference");
        
        println!("\nInference summary:");
        println!("- Input token: {}", token_id);
        println!("- Processed {} transformer blocks", n_blocks);
        println!("- Final logits shape: {}", logits.len());
        println!("- First few logits: {:?}", &logits[..5.min(logits.len())]);
        
        Ok(logits)
    }

    /// Projects hidden state to vocabulary logits
    /// 
    /// # Design Notes
    /// Final projection matrix shape: [hidden_dim, vocab_size] (2048, 32000)
    /// Input: [hidden_dim] (2048)
    /// Output: [vocab_size] (32000)
    fn project_to_vocab(&self, hidden_state: &[f32]) -> io::Result<Vec<f32>> {
        let output_idx = self.tensors.iter()
            .position(|t| t.name == "output.weight")
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Output weights not found"))?;
            
        println!("\nFinal projection to vocabulary:");
        println!("- Hidden state size: {}", hidden_state.len());
        println!("- Output matrix shape: {:?}", self.tensors[output_idx].dimensions);
        
        // Use transpose=true for [2048, 32000] × [2048] -> [32000]
        let logits = self.matmul_q8_0_vec(output_idx, hidden_state, true)?;
        
        println!("- Output logits size: {}", logits.len());
        
        Ok(logits)
    }

    /// Implements the final projection to vocabulary logits
    /// 
    /// # Design Notes
    /// The final layer projects from model dimension to vocabulary size:
    /// - Input: [hidden_dim] (2048)
    /// - Output weight: [hidden_dim, vocab_size] (2048, 32000)
    /// - Output: [vocab_size] (32000)
    /// 
    /// Memory efficiency:
    /// - Only loads the final projection matrix when needed
    /// - Processes one row at a time if needed
    fn final_projection(&self, hidden_state: &[f32]) -> io::Result<Vec<f32>> {
        let output_idx = self.tensors.iter()
            .position(|t| t.name == "output.weight")
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Output weights not found"))?;
            
        // For final projection, we want matrix[2048, 32000] × input[2048] -> output[32000]
        // So we use transpose=true
        let logits = self.matmul_q8_0_vec(output_idx, hidden_state, true)?;
        
        println!("Final projection:");
        println!("- Input size: {}", hidden_state.len());
        println!("- Output size: {}", logits.len());
        println!("- First few logits: {:?}", &logits[..5.min(logits.len())]);
        
        Ok(logits)
    }

    /// Applies final layer normalization before output projection.
    fn apply_final_norm(&self, norm_idx: usize, input: &[f32]) -> io::Result<Vec<f32>> {
        let norm_data = self.get_tensor_data(norm_idx)?;
        
        let mut sum = 0.0f32;
        for &x in input {
            sum += x * x;
        }
        let rms = (sum / input.len() as f32).sqrt();
        
        let mut normalized = vec![0.0; input.len()];
        for i in 0..input.len() {
            normalized[i] = input[i] / (rms + 1e-5) * 
                (norm_data[i] as i8 as f32);
        }
        
        Ok(normalized)
    }  

    /// Implements a full transformer block, combining attention and feed-forward networks.
    /// 
    /// # Design Notes
    /// Transformer block processing order:
    /// 1. Input -> Attention -> Residual connection
    /// 2. Result -> FFN -> Residual connection
    /// 
    /// Memory efficiency maintained by:
    /// - Processing sequentially
    /// - Adding residual connections in-place
    /// - Reusing vectors where possible
    /// 
    /// # Arguments
    /// * `block` - Block index in the transformer
    /// * `input` - Input tensor
    /// 
    /// # Returns
    /// * `Result<Vec<f32>>` - Block output tensor of same shape as input
    fn transformer_block(&self, block: usize, input: &[f32]) -> io::Result<Vec<f32>> {
        self.log_memory("Start transformer block");
        
        // 1. Attention with residual connection
        let attention_out = self.get_attention_layer(block, input)?;
        let mut residual = vec![0.0; input.len()];
        for i in 0..input.len() {
            residual[i] = attention_out[i] + input[i];
        }
        
        self.log_memory("After attention residual");
        
        // 2. Feed-forward with residual connection
        let ffn_out = self.feed_forward(block, &residual)?;
        for i in 0..input.len() {
            residual[i] = ffn_out[i] + residual[i];
        }
        
        self.log_memory("End transformer block");
        
        println!("Transformer block {} sample values:", block);
        println!("- Input first value: {}", input[0]);
        println!("- After attention: {}", attention_out[0]);
        println!("- After FFN: {}", ffn_out[0]);
        println!("- Final output: {}", residual[0]);
        
        Ok(residual)
    }
    
    /// Implements feed-forward network (FFN) for a transformer block
    /// 
    /// # Design Notes
    /// The FFN consists of three operations:
    /// 1. Up projection:   input[2048] -> hidden[5632]
    /// 2. Gate activation: input[2048] -> hidden[5632]
    /// 3. Down projection: hidden[5632] -> output[2048]
    /// 
    /// Matrix dimensions are critical:
    /// - Up/Gate: [2048, 5632] × [2048] -> [5632]
    /// - Down: [5632, 2048] × [5632] -> [2048]
    /// 
    /// # Arguments
    /// * `block` - Block index in transformer
    /// * `input` - Input tensor [2048]
    /// 
    /// # Memory Usage
    /// Peaks during up-projection due to larger hidden dimension
    fn feed_forward(&self, block: usize, input: &[f32]) -> io::Result<Vec<f32>> {
        // Get FFN dimensions from metadata
        let hidden_dim = match &self.header.metadata.kvs.get("llama.feed_forward_length") {
            Some(MetadataValue::U64(dim)) => *dim as usize,
            Some(MetadataValue::U32(dim)) => *dim as usize,
            _ => return Err(io::Error::new(io::ErrorKind::NotFound, "FFN hidden dimension not found"))
        };

        println!("FFN dimensions:");
        println!("- Input dim: {}", input.len());
        println!("- Hidden dim: {}", hidden_dim);

        // Get weight matrices
        let up_idx = self.tensors.iter()
            .position(|t| t.name == format!("blk.{}.ffn_up.weight", block))
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "FFN up weights not found"))?;
            
        let down_idx = self.tensors.iter()
            .position(|t| t.name == format!("blk.{}.ffn_down.weight", block))
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "FFN down weights not found"))?;

        let gate_idx = self.tensors.iter()
            .position(|t| t.name == format!("blk.{}.ffn_gate.weight", block))
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "FFN gate weights not found"))?;

        let up_tensor = &self.tensors[up_idx];
        let down_tensor = &self.tensors[down_idx];
        println!("FFN matrix shapes:");
        println!("- Up weight: {:?}", up_tensor.dimensions);
        println!("- Down weight: {:?}", down_tensor.dimensions);

        // First normalize input
        let normalized = self.apply_ffn_norm(block, input)?;
        println!("Normalized input length: {}", normalized.len());
        
        self.log_memory("Before FFN up-projection");
        
        // Up projection: [2048, 5632] × [2048] -> [5632]
        println!("\nAttempting up projection:");
        println!("Matrix: [{:?}] × input[{}]", up_tensor.dimensions, normalized.len());
        let up = self.matmul_q8_0_vec(up_idx, &normalized, true)?;
        println!("Up projection successful, length: {}", up.len());
        
        // Gate projection
        println!("\nAttempting gate projection:");
        let gate = self.matmul_q8_0_vec(gate_idx, &normalized, true)?;
        println!("Gate projection successful, length: {}", gate.len());
        
        // Apply activation
        let mut activated = Vec::with_capacity(up.len());
        for i in 0..up.len() {
            let sigmoid_gate = 1.0 / (1.0 + (-gate[i]).exp());
            activated.push(up[i] * sigmoid_gate);
        }
        println!("Activation successful, length: {}", activated.len());
        
        self.log_memory("Before FFN down-projection");
        
        // // Down projection: [5632, 2048] × [5632] -> [2048]
        // println!("\nAttempting down projection:");
        // println!("Matrix: [{:?}] × input[{}]", down_tensor.dimensions, activated.len());
        // let down = self.matmul_q8_0_vec(down_idx, &activated, false)?;
        // println!("Down projection successful, length: {}", down.len());
        
        // Down projection: [5632, 2048] × [5632] -> [2048]
        println!("\nAttempting down projection:");
        println!("Matrix: [{:?}] × input[{}]", down_tensor.dimensions, activated.len());
        // Use transpose=true because we want to multiply [5632] × [2048, 5632]^T -> [2048]
        let down = self.matmul_q8_0_vec(down_idx, &activated, true)?;
        println!("Down projection successful, length: {}", down.len());
            
        self.log_memory("After FFN");
        
        println!("FFN sample values:");
        println!("- Input first value: {}", input[0]);
        println!("- Up-projection first value: {}", up[0]);
        println!("- Gate first value: {}", gate[0]);
        println!("- Activated first value: {}", activated[0]);
        println!("- Output first value: {}", down[0]);
        
        Ok(down)
    }   

    /// Applies layer normalization for feed-forward network input.
    /// Similar to attention normalization but uses FFN-specific weights.
    fn apply_ffn_norm(&self, block: usize, input: &[f32]) -> io::Result<Vec<f32>> {
        let norm_idx = self.tensors.iter()
            .position(|t| t.name == format!("blk.{}.ffn_norm.weight", block))
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "FFN norm weights not found"))?;
                    
        let norm_data = self.get_tensor_data(norm_idx)?;
        
        let mut sum = 0.0f32;
        for &x in input {
            sum += x * x;
        }
        let rms = (sum / input.len() as f32).sqrt();
        
        let mut normalized = vec![0.0; input.len()];
        for i in 0..input.len() {
            normalized[i] = input[i] / (rms + 1e-5) * 
                (norm_data[i] as i8 as f32);
        }
        
        Ok(normalized)
    }

    fn apply_attention_norm(&self, block: usize, input: &[f32]) -> io::Result<Vec<f32>> {
        // Get norm weights
        let norm_idx = self.tensors.iter()
            .position(|t| t.name == format!("blk.{}.attn_norm.weight", block))
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, 
                "Norm weights not found"))?;
                
        let norm_data = self.get_tensor_data(norm_idx)?;
        
        // Basic RMSNorm implementation
        let mut sum = 0.0f32;
        for &x in input {
            sum += x * x;
        }
        let rms = (sum / input.len() as f32).sqrt();
        
        // Apply normalization
        let mut normalized = vec![0.0; input.len()];
        for i in 0..input.len() {
            normalized[i] = input[i] / (rms + 1e-5) * 
                (norm_data[i] as i8 as f32);
        }
        
        Ok(normalized)
    }

    fn calculate_qkv(&self, block: usize, input: &[f32]) -> io::Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let qkv_idx = self.tensors.iter()
            .position(|t| t.name == format!("blk.{}.attn_qkv.weight", block))
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "QKV matrix not found"))?;
            
        let qkv_tensor = &self.tensors[qkv_idx];
        let hidden_size = input.len();
        
        // For now just split the QKV matrix into thirds
        // This is simplified - actual implementation needs to handle head dimension properly
        let qkv_data = self.get_q8_0_tensor(qkv_idx)?;
        let third = qkv_data.len() / 3;
        
        let q = qkv_data[0..third].to_vec();
        let k = qkv_data[third..2*third].to_vec();
        let v = qkv_data[2*third..].to_vec();
        
        Ok((q, k, v))
    }
    
    fn get_f32_tensor(&self, tensor_idx: usize) -> io::Result<Vec<f32>> {
        let tensor = &self.tensors[tensor_idx];
        if let GGMLType::F32 = tensor.tensor_type {
            let data = self.get_tensor_data(tensor_idx)?;
            let mut result = Vec::with_capacity(tensor.element_count() as usize);
            
            for chunk in data.chunks(4) {
                if chunk.len() == 4 {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    result.push(value);
                }
            }
            
            Ok(result)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Tensor is not F32 type"
            ))
        }
    }
            
    /// Performs matrix-vector multiplication with support for transposed matrices.
    /// 
    /// # Design Notes
    /// Matrix multiplication cases we need to handle:
    /// 1. Regular (transpose=false):
    ///    - Matrix shape [M, N] × input[N] -> output[M]
    ///    - Example: [5632, 2048] × [2048] -> [5632]
    /// 2. Transposed (transpose=true):
    ///    - Matrix shape [N, M] × input[N] -> output[M]
    ///    - Example: [2048, 5632] × [2048] -> [5632]
    /// 
    /// # Matrix Layout
    /// - Regular: matrix[i][j] = data[i * cols + j]
    /// - Transposed: matrix[i][j] = data[j * rows + i]
    /// 
    /// # Dimension Checking
    /// - For regular: input.len() must match matrix.dimensions[1]
    /// - For transposed: input.len() must match matrix.dimensions[0]
    /// 
    /// # Arguments
    /// * `matrix_idx` - Index of weight matrix in tensors
    /// * `vec` - Input vector
    /// * `transpose` - Whether to treat matrix as transposed
    /// 
    /// # Returns
    /// * `Result<Vec<f32>>` - Output vector
    fn matmul_q8_0_vec(&self, matrix_idx: usize, vec: &[f32], transpose: bool) -> io::Result<Vec<f32>> {
        let matrix = &self.tensors[matrix_idx];
        
        // Get matrix dimensions based on transpose flag
        let (rows, cols) = if transpose {
            // For transpose, we use dimensions in reverse
            (matrix.dimensions[1] as usize, matrix.dimensions[0] as usize)
        } else {
            (matrix.dimensions[0] as usize, matrix.dimensions[1] as usize)
        };
        
        // Input dimension check
        let expected_input_len = if transpose { matrix.dimensions[0] } else { matrix.dimensions[1] } as usize;
        if vec.len() != expected_input_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Input length mismatch: expected {}, got {}\nMatrix shape: {:?}, transpose: {}", 
                    expected_input_len, vec.len(), matrix.dimensions, transpose)
            ));
        }

        println!("Matrix multiplication:");
        println!("- Matrix shape: {:?}", matrix.dimensions);
        println!("- Transpose: {}", transpose);
        println!("- Input length: {}", vec.len());
        println!("- Expected output length: {}", rows);

        // Get matrix data
        let matrix_data = self.get_q8_0_tensor(matrix_idx)?;
        let mut result = vec![0.0f32; rows];
        
        // Compute matrix-vector product
        for i in 0..rows {
            let mut sum = 0.0f32;
            for j in 0..cols {
                // Index calculation differs for transposed matrix
                let matrix_idx = if transpose {
                    j * matrix.dimensions[1] as usize + i  // Transposed access
                } else {
                    i * cols + j  // Regular access
                };
                sum += matrix_data[matrix_idx] * vec[j];
            }
            result[i] = sum;
        }

        Ok(result)
    }

    /// Implements multi-head attention with grouped-query attention (GQA) support.
    /// Processes Q-K-V interaction one head at a time to minimize memory usage.
    /// 
    /// # Arguments
    /// * `q` - Query tensor after projection, shape [batch_size, n_heads * head_size]
    /// * `k` - Key tensor after projection, shape [batch_size, n_kv_heads * head_size]
    /// * `v` - Value tensor after projection, shape [batch_size, n_kv_heads * head_size]
    /// * `n_heads` - Number of query attention heads
    /// 
    /// # Returns
    /// * `Result<Vec<f32>>` - Attention output tensor of same shape as query input
    /// 
    /// # Memory Usage
    /// Processes one head at a time, allocating:
    /// - One vector for attention scores (head_size)
    /// - Output vector (same size as input q)
    /// Implements multi-head attention with grouped-query attention (GQA) support.
    /// Includes input normalization, scaling, and softmax to maintain numerical stability.
    /// Implements multi-head attention with grouped-query attention (GQA) support.
    /// 
    /// # Design Notes
    /// The implementation handles numerical stability through several mechanisms:
    /// 1. Input Normalization: Q, K, V values are normalized by their respective max absolute values
    ///    to prevent numerical explosion during matrix multiplication while preserving relative relationships
    /// 2. Attention Scaling: Uses 1/√(head_size) scaling factor as per "Attention Is All You Need"
    /// 3. Stable Softmax: Implements numerically stable softmax by subtracting max value before exp
    /// 4. Output Rescaling: Results are rescaled back using V's max value to maintain original scale
    /// 
    /// # Memory Management
    /// - Processes one head at a time to minimize memory footprint
    /// - Allocates temporary vectors for normalized values and scores within head scope
    /// - Peak memory usage occurs during score calculation (head_size * kv_head_size)
    /// 
    /// # Numerical Stability Strategy
    /// Initial tests showed:
    /// 1. Without normalization: values exploded to 56207075000000
    /// 2. With only scaling factor: values underflowed to 0
    /// 3. Current approach with normalization: values stay in reasonable range (-600 to -500)
    /// 
    /// # Arguments
    /// * `q` - Query tensor [batch_size, n_heads * head_size]
    /// * `k` - Key tensor [batch_size, n_kv_heads * head_size]
    /// * `v` - Value tensor [batch_size, n_kv_heads * head_size]
    /// * `n_heads` - Number of query attention heads
    /// 
    /// # Returns
    /// * `Result<Vec<f32>>` - Attention output tensor of same shape as query input
    fn attention_qkv(&self, q: &[f32], k: &[f32], v: &[f32], n_heads: usize) -> io::Result<Vec<f32>> {
        /*
        Key changes:

        Normalize Q, K, and V values independently using their max absolute values
        Added small epsilon (1e-5) to prevent division by zero
        Rescale output back to original scale using V's max value
        Added more detailed debugging output to track normalization
        
        The goal is to:
        
        Keep values in a reasonable range during matrix multiplication
        Preserve relative relationships between values
        Output results in a similar scale to the input
        
        The docstring now captures:
        
        The evolution of the implementation and why we made certain choices
        Memory management strategy
        Numerical stability approaches and their results
        Detailed explanation of the normalization strategy
                
        */
        let head_size = q.len() / n_heads;
        let kv_head_size = k.len() / n_heads;
        
        println!("Attention QKV dimensions:");
        println!("- Total size: Q={}, K={}, V={}", q.len(), k.len(), v.len());
        println!("- Head size: Q={}, KV={}", head_size, kv_head_size);
        println!("- Number of heads: {}", n_heads);
        
        let mut result = vec![0.0; q.len()];
        
        // Scaling factor for attention scores
        let scaling_factor = 1.0 / (head_size as f32).sqrt();
        
        for head in 0..n_heads {
            let q_start = head * head_size;
            let q_end = q_start + head_size;
            let q_head = &q[q_start..q_end];
            
            let kv_head_idx = head * n_heads / k.len();
            let kv_start = kv_head_idx * kv_head_size;
            let kv_end = kv_start + kv_head_size;
            
            let k_head = &k[kv_start..kv_end];
            let v_head = &v[kv_start..kv_end];
            
            // Normalize Q and K values before multiplication
            let q_max = q_head.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            let k_max = k_head.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            
            let q_norm: Vec<f32> = q_head.iter().map(|&x| x / (q_max + 1e-5)).collect();
            let k_norm: Vec<f32> = k_head.iter().map(|&x| x / (k_max + 1e-5)).collect();
            
            // Calculate scaled attention scores
            let mut scores = Vec::with_capacity(head_size);
            for i in 0..head_size {
                let mut score = 0.0;
                for j in 0..kv_head_size {
                    score += q_norm[i] * k_norm[j] * scaling_factor;
                }
                scores.push(score);
            }
            
            // Apply stable softmax to scores
            let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_scores: Vec<f32> = scores.iter()
                .map(|&s| ((s - max_score).exp()))
                .collect();
            let exp_sum: f32 = exp_scores.iter().sum();
            for score in &mut exp_scores {
                *score /= exp_sum;
            }
            
            // Normalize V values
            let v_max = v_head.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            let v_norm: Vec<f32> = v_head.iter().map(|&x| x / (v_max + 1e-5)).collect();
            
            // Apply normalized attention scores to normalized values
            for i in 0..head_size {
                let mut weighted_sum = 0.0;
                for j in 0..kv_head_size {
                    weighted_sum += exp_scores[i] * v_norm[j];
                }
                // Rescale output back to original scale
                result[q_start + i] = weighted_sum * v_max;
            }
            
            if head == 0 {
                println!("First head sample values:");
                println!("- Q max: {}, first normalized: {}", q_max, q_norm[0]);
                println!("- K max: {}, first normalized: {}", k_max, k_norm[0]);
                println!("- V max: {}, first normalized: {}", v_max, v_norm[0]);
                println!("- Raw score first value: {}", scores[0]);
                println!("- Normalized score first value: {}", exp_scores[0]);
                println!("- Result first value: {}", result[q_start]);
            }
        }
        
        Ok(result)
    }
        
    /// Processes a single attention layer in a transformer block.
    /// Handles the full attention mechanism including input normalization,
    /// Q-K-V projections, and multi-head attention with GQA support.
    /// 
    /// # Arguments
    /// * `block` - Block index in the transformer
    /// * `input` - Input tensor
    /// 
    /// # Returns
    /// * `Result<Vec<f32>>` - Attention mechanism output
    /// 
    /// # Memory Usage
    /// Peak memory usage occurs during Q-K-V projections and attention calculation
    fn get_attention_layer(&self, block: usize, input: &[f32]) -> io::Result<Vec<f32>> {
        self.log_memory("Start attention calculation");
        
        let n_heads = match &self.header.metadata.kvs.get("llama.attention.head_count") {
            Some(MetadataValue::U64(heads)) => *heads as usize,
            Some(MetadataValue::U32(heads)) => *heads as usize,
            _ => return Err(io::Error::new(io::ErrorKind::NotFound, "Head count not found"))
        };

        // Get Q, K, V matrices
        let q_idx = self.tensors.iter()
            .position(|t| t.name == format!("blk.{}.attn_q.weight", block))
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Q matrix not found"))?;
            
        let k_idx = self.tensors.iter()
            .position(|t| t.name == format!("blk.{}.attn_k.weight", block))
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "K matrix not found"))?;
            
        let v_idx = self.tensors.iter()
            .position(|t| t.name == format!("blk.{}.attn_v.weight", block))
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "V matrix not found"))?;

        let q_tensor = &self.tensors[q_idx];
        let k_tensor = &self.tensors[k_idx];
        let v_tensor = &self.tensors[v_idx];
        
        println!("\nTensor shapes:");
        println!("Q: {:?}", q_tensor.dimensions);
        println!("K: {:?}", k_tensor.dimensions);
        println!("V: {:?}", v_tensor.dimensions);
        println!("Input length: {}", input.len());

        // First normalize input
        let normalized = self.apply_attention_norm(block, input)?;
        println!("Normalized input length: {}", normalized.len());
        
        println!("\nAttempting matrix multiplications:");
        println!("Q matrix: [{}x{}] × input[{}]", q_tensor.dimensions[0], q_tensor.dimensions[1], normalized.len());
        let q = self.matmul_q8_0_vec(q_idx, &normalized, false)?;
        println!("Q projection successful, length: {}", q.len());

        println!("K matrix: [{}x{}] × input[{}]", k_tensor.dimensions[0], k_tensor.dimensions[1], normalized.len());
        let k = self.matmul_q8_0_vec(k_idx, &normalized, true)?;  // Try transpose=true for K
        println!("K projection successful, length: {}", k.len());

        println!("V matrix: [{}x{}] × input[{}]", v_tensor.dimensions[0], v_tensor.dimensions[1], normalized.len());
        let v = self.matmul_q8_0_vec(v_idx, &normalized, true)?;  // Try transpose=true for V
        
        println!("\nProjection lengths:");
        println!("Q: {}", q.len());
        println!("K: {}", k.len());
        println!("V: {}", v.len());
        
        // Apply attention mechanism
        let attention_output = self.attention_qkv(&q, &k, &v, n_heads)?;
        
        self.log_memory("End attention calculation");
        Ok(attention_output)
    }

    fn get_q8_0_tensor(&self, tensor_idx: usize) -> io::Result<Vec<f32>> {
        let tensor = &self.tensors[tensor_idx];
        if let GGMLType::Q8_0 = tensor.tensor_type {
            let data = self.get_tensor_data(tensor_idx)?;
            let n = tensor.element_count() as usize;
            
            // Simple 1:1 conversion of bytes to f32
            let result: Vec<f32> = data.iter()
                .map(|&x| x as i8 as f32)
                .collect();
                
            Ok(result)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Tensor is not Q8_0 type"
            ))
        }
    }

    fn log_memory(&self, operation: &str) {
        if let Ok(proc_self) = std::fs::read_to_string("/proc/self/status") {
            if let Some(line) = proc_self.lines().find(|l| l.starts_with("VmRSS:")) {
                println!("Memory usage at {}: {}", operation, line);
            }
        }
    }

    
    fn get_token_embedding(&self, token_id: usize) -> io::Result<Vec<f32>> {
        let embed_idx = self.tensors.iter()
            .position(|t| t.name == "token_embd.weight")
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Token embedding matrix not found"))?;
    
        self.log_memory("Before reading embedding");
        
        let tensor = &self.tensors[embed_idx];
        let embedding_dim = tensor.dimensions[0] as usize;
        let vocab_size = tensor.dimensions[1] as usize;
    
        if token_id >= vocab_size {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Token ID out of range"));
        }

        // Calculate offset for just this token's embedding
        let data = self.get_tensor_data(embed_idx)?;
        let row_size = embedding_dim; // For Q8_0, one byte per value
        let start = token_id * row_size;
        let row_data = &data[start..start + row_size];
    
        // Convert just this row from Q8_0 to f32
        let embedding: Vec<f32> = row_data.iter()
            .map(|&x| x as i8 as f32)
            .collect();
    
        self.log_memory("After reading embedding");
    
        Ok(embedding)
    }
}

/// Calculates theoretical memory requirements for model
/// 
/// # Arguments
/// * `arch` - Model architecture parameters
/// 
/// # Returns
/// * Memory requirements in different configurations
struct MemoryRequirements {
    minimum_mb: f64,    // Minimum working memory needed
    full_model_mb: f64, // Full model loaded into memory
    per_layer_mb: f64,  // Memory needed per transformer layer
}

fn calculate_memory_requirements(arch: &ModelArchitecture) -> MemoryRequirements {
    // Calculate sizes in bytes
    let hidden_size_bytes = arch.hidden_size * std::mem::size_of::<f32>();
    let vocab_size_bytes = arch.vocab_size * std::mem::size_of::<f32>();
    
    // Per-layer components
    let attention_bytes = hidden_size_bytes * arch.hidden_size * 3; // Q,K,V
    let ffn_bytes = hidden_size_bytes * arch.intermediate_size * 2; // Up,Down
    let layer_bytes = attention_bytes + ffn_bytes;
    
    // Convert to MB
    let mb = |bytes: usize| (bytes as f64) / (1024.0 * 1024.0);
    
    MemoryRequirements {
        minimum_mb: mb(hidden_size_bytes * 4), // Minimum working buffers
        full_model_mb: mb(layer_bytes * arch.num_layers + vocab_size_bytes),
        per_layer_mb: mb(layer_bytes),
    }
}

/// Main entry point with command-line options for model inspection or inference
/// 
/// # Usage Examples
/// ```bash
/// # Inspect model architecture
/// cargo run -- --model REAL_PATH/model.gguf --inspect
/// 
/// # Run inference
/// cargo run -- --model REAL_PATH/model.gguf --inference --token 1
///
/// # Both inspect and run inference
/// cargo run -- --model models/llamacorn-1.1b/llamacorn-1.1b-chat.Q8_0.gguf --inspect --inference
///
/// e.g. 
/// # Inspect model architecture
/// cargo run -- --model /home/oopsy/jan/models/llamacorn-1.1b/llamacorn-1.1b-chat.Q8_0.gguf --inspect
/// 
/// # Run inference
/// cargo run -- --model /home/oopsy/jan/models/llamacorn-1.1b/llamacorn-1.1b-chat.Q8_0.gguf --inference --token 1
/// cargo run -- --model /home/oopsy/jan/models/llamacorn-1.1b/llamacorn-1.1b-chat.Q8_0.gguf--inference --text "Hello world"
///
/// ```
fn main() -> io::Result<()> {

    // Get command line arguments
    let args: Vec<String> = std::env::args().collect();
    
    // Simple arg parsing
    let mut model_path = None;
    let mut should_inspect = false;
    let mut should_infer = false;
    let mut token_id = 1;
    
    // TODO clarify, maybe this is to preset/reset a default value
    let mut input_text = String::from("Hello");  // default
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                if i + 1 < args.len() {
                    model_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    println!("Error: --model requires a path argument");
                    return Ok(());
                }
            },
            "--inspect" => {
                should_inspect = true;
                i += 1;
            },
            "--inference" => {
                should_infer = true;
                i += 1;
            },
            "--token" => {
                if i + 1 < args.len() {
                    token_id = args[i + 1].parse().unwrap_or(1);
                    i += 2;
                } else {
                    println!("Error: --token requires a number argument");
                    return Ok(());
                }
            },
            "--text" => {
                if i + 1 < args.len() {
                    input_text = args[i + 1].clone();
                    i += 2;
                } else {
                    println!("Error: --text requires a string argument");
                    return Ok(());
                }
            },
            _ => {
                println!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    // Verify we have a model path
    let model_path = model_path.ok_or_else(|| {
        println!("Usage:");
        println!("  Inspect model:  --model <path> --inspect");
        println!("  Run inference:  --model <path> --inference [--token <id>]");
        io::Error::new(io::ErrorKind::InvalidInput, "No model path provided")
    })?;

    // Load the model
    println!("Loading model from: {}", model_path);
    let model = GGUFModel::load(&model_path)?;

    if should_inspect {
        println!("\nGGUF Model Information:");
        println!("Version: {}", model.header.version);
        println!("Number of tensors: {}", model.header.tensor_count);
        
        // Run architecture inspection
        println!("\nModel Architecture:");
        let arch = model.inspect_architecture()?;
        
        // Add tokenizer inspection here
        println!("\nTokenizer Information:");
        model.inspect_tokenizer()?;
        
        // Print memory requirements
        let theoretical_memory = calculate_memory_requirements(&arch);
        println!("\nTheoretical Memory Requirements:");
        println!("- Minimum working set: {} MB", theoretical_memory.minimum_mb);
        println!("- Full model size: {} MB", theoretical_memory.full_model_mb);
        println!("- Per-layer memory: {} MB", theoretical_memory.per_layer_mb);
        
        // Print first few tensors
        println!("\nFirst 5 tensors:");
        for (i, tensor) in model.tensors.iter().enumerate().take(5) {
            println!("Name: {}", tensor.name);
            println!("  Dimensions: {:?}", tensor.dimensions);
            println!("  Type: {:?}", tensor.tensor_type);
            println!("  Size: {:.2} MB", tensor.byte_size() as f64 / (1024.0 * 1024.0));
            println!("  Offset: {}", tensor.offset);
            
            // Show first few bytes of tensor data
            if let Ok(data) = model.get_tensor_data(i) {
                println!("  First few bytes: {:?}", data.get(..16));
            }
        }
    }
    
    // if should_infer {
    //     // Run inference
    //     println!("\nRunning inference with token {}...", token_id);
    //     match model.run_model_inference(token_id) {
    //         Ok(logits) => {
    //             println!("\nSuccessfully computed logits");
    //             println!("Top 5 next token probabilities:");
                
    //             let mut indices: Vec<usize> = (0..logits.len()).collect();
    //             indices.sort_by(|&i, &j| logits[j].partial_cmp(&logits[i])
    //                 .unwrap_or(std::cmp::Ordering::Equal));
                
    //             for &idx in indices.iter().take(5) {
    //                 println!("Token {}: {:.3}", idx, logits[idx]);
    //             }
    //         },
    //         Err(e) => println!("Error in inference: {}", e),
    //     }
    // }
    if should_infer {
        println!("\nRunning string inference...");
        match model.infer_string(&input_text) {
            Ok(output) => {
                println!("Input text: {}", input_text);
                println!("Output text: {}", output);
            },
            Err(e) => println!("Error in string inference: {}", e),
        }
    }
    
    
    Ok(())
}
