// Constants for model configuration
const CONFIG = {
  numLayers: 4,           // จำนวน transformer layers
  numHeads: 4,           // จำนวน attention heads
  embedDim: 256,         // ขนาดของ embedding vector
  ffnDim: 512,          // ขนาดของ feed-forward network
  maxSeqLength: 128,     // ความยาวสูงสุดของ sequence
  vocabSize: 128,        // ขนาดของ vocabulary
  dropout: 0.1          // อัตรา dropout
};

// -------------------------------
// Positional Encoding
// -------------------------------
function getPositionalEncoding(seqLength, embedDim) {
  const pe = Array(seqLength).fill().map(() => Array(embedDim).fill(0));
  
  for (let pos = 0; pos < seqLength; pos++) {
    for (let i = 0; i < embedDim; i += 2) {
      const freq = Math.pow(10000, -i / embedDim);
      pe[pos][i] = Math.sin(pos * freq);
      if (i + 1 < embedDim) {
        pe[pos][i + 1] = Math.cos(pos * freq);
      }
    }
  }
  return pe;
}

// -------------------------------
// Multi-Head Self-Attention
// -------------------------------
function multiHeadSelfAttention(Q, K, V, numHeads) {
  const d_k = Math.floor(Q[0].length / numHeads);
  let outputs = [];

  // แบ่ง input สำหรับแต่ละ head
  for (let h = 0; h < numHeads; h++) {
    const Q_h = Q.map(row => row.slice(h * d_k, (h + 1) * d_k));
    const K_h = K.map(row => row.slice(h * d_k, (h + 1) * d_k));
    const V_h = V.map(row => row.slice(h * d_k, (h + 1) * d_k));

    // คำนวณ attention สำหรับแต่ละ head
    const scores = matrixMultiply(Q_h, transpose(K_h));
    const scaledScores = scores.map(row => 
      row.map(value => value / Math.sqrt(d_k))
    );
    
    // Apply mask for causal attention (ป้องกันการมองไปข้างหน้า)
    for (let i = 0; i < scaledScores.length; i++) {
      for (let j = i + 1; j < scaledScores[i].length; j++) {
        scaledScores[i][j] = -Infinity;
      }
    }

    const attentionWeights = scaledScores.map(row => softmax(row));
    const headOutput = matrixMultiply(attentionWeights, V_h);
    outputs.push(headOutput);
  }

  // Concatenate all heads
  return outputs.reduce((acc, head) => {
    return acc.map((row, i) => [...row, ...head[i]]);
  }, outputs[0].map(() => []));
}

// -------------------------------
// Feed Forward Network
// -------------------------------
function feedForward(input, ffnDim) {
  // First linear transformation
  const W1 = createRandomMatrix(input[0].length, ffnDim);
  const b1 = Array(ffnDim).fill(0);
  let hidden = matrixMultiply(input, W1);
  hidden = hidden.map(row => row.map((val, j) => val + b1[j]));
  
  // ReLU activation
  hidden = hidden.map(row => row.map(val => Math.max(0, val)));
  
  // Second linear transformation
  const W2 = createRandomMatrix(ffnDim, input[0].length);
  const b2 = Array(input[0].length).fill(0);
  let output = matrixMultiply(hidden, W2);
  output = output.map(row => row.map((val, j) => val + b2[j]));
  
  return output;
}

// -------------------------------
// Layer Normalization
// -------------------------------
function layerNorm(input, epsilon = 1e-5) {
  return input.map(row => {
    const mean = row.reduce((a, b) => a + b) / row.length;
    const variance = row.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / row.length;
    return row.map(val => (val - mean) / Math.sqrt(variance + epsilon));
  });
}

// -------------------------------
// Transformer Layer
// -------------------------------
function transformerLayer(input, config) {
  // Multi-head self-attention
  let attentionOutput = multiHeadSelfAttention(
    input, input, input,
    config.numHeads
  );
  
  // Add & Norm
  attentionOutput = matrixAdd(input, attentionOutput);
  attentionOutput = layerNorm(attentionOutput);
  
  // Feed-forward network
  let ffnOutput = feedForward(attentionOutput, config.ffnDim);
  
  // Add & Norm
  ffnOutput = matrixAdd(attentionOutput, ffnOutput);
  ffnOutput = layerNorm(ffnOutput);
  
  // Apply dropout
  ffnOutput = applyDropout(ffnOutput, config.dropout);
  
  return ffnOutput;
}

// -------------------------------
// Token Embedding
// -------------------------------
function getTokenEmbeddings(tokens, embedDim) {
  const embeddings = createRandomMatrix(CONFIG.vocabSize, embedDim);
  return tokens.map(token => embeddings[token % CONFIG.vocabSize]);
}

// -------------------------------
// Main Model Function
// -------------------------------
function thaiLanguageModel(input, config) {
  // Convert input to tokens
  let tokens = utf8ToTokens(input);
  if (tokens.length > config.maxSeqLength) {
    tokens = tokens.slice(0, config.maxSeqLength);
  }
  
  // Get embeddings
  let embedded = getTokenEmbeddings(tokens, config.embedDim);
  
  // Add positional encoding
  const posEncoding = getPositionalEncoding(tokens.length, config.embedDim);
  embedded = matrixAdd(embedded, posEncoding);
  
  // Pass through transformer layers
  let output = embedded;
  for (let i = 0; i < config.numLayers; i++) {
    output = transformerLayer(output, config);
  }
  
  // Final layer normalization
  output = layerNorm(output);
  
  // Project to vocabulary space and apply softmax
  const logits = projectToVocab(output, config.vocabSize);
  const probabilities = logits.map(row => softmax(row));
  
  return probabilities;
}

// -------------------------------
// Helper Functions
// -------------------------------
function matrixMultiply(A, B) {
  if (A[0].length !== B.length) {
    throw new Error('Matrix dimensions do not match for multiplication');
  }
  const result = Array(A.length).fill().map(() => Array(B[0].length).fill(0));
  
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < B[0].length; j++) {
      for (let k = 0; k < B.length; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return result;
}

function transpose(matrix) {
  return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
}

function softmax(values) {
  const maxVal = Math.max(...values);
  const expValues = values.map(val => Math.exp(val - maxVal));
  const sumExp = expValues.reduce((acc, val) => acc + val, 0);
  return expValues.map(val => val / sumExp);
}

function createRandomMatrix(rows, cols) {
  return Array(rows).fill().map(() => 
    Array(cols).fill().map(() => (Math.random() - 0.5) * 2 / Math.sqrt(cols))
  );
}

function matrixAdd(A, B) {
  return A.map((row, i) => row.map((val, j) => val + B[i][j]));
}

function applyDropout(matrix, rate) {
  return matrix.map(row => 
    row.map(val => Math.random() > rate ? val / (1 - rate) : 0)
  );
}

function projectToVocab(hidden, vocabSize) {
  const projectionMatrix = createRandomMatrix(hidden[0].length, vocabSize);
  return matrixMultiply(hidden, projectionMatrix);
}

function utf8ToTokens(str) {
  return Array.from(str).map(char => char.charCodeAt(0));
}

// Example usage
const input = "สวัสดี! มีอะไรให้ฉันช่วยไหม?";
const output = thaiLanguageModel(input, CONFIG);
console.log("Model output probabilities:", output);