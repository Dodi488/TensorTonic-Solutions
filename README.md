# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Rodrigo García Martínez (Rodrigo G.)'s TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/dodi488.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Implement Adam Optimizer Step | Implement one vectorized Adam optimizer step in NumPy with first and second moments, bias correction, and elementwise parameter updates. | https://www.tensortonic.com/problems/adam-optimizer |
| Implement AdamW (Decoupled Weight Decay) | Implement one AdamW optimizer step in NumPy with first and second moments plus decoupled weight decay. | https://www.tensortonic.com/problems/adamw-optimizer |
| Argmax | Implement a parallel CUDA argmax reduction that returns the lowest index when multiple elements share the maximum. | https://www.tensortonic.com/problems/argmax |
| Argmin | Implement a parallel CUDA argmin reduction that returns the lowest index when multiple elements share the minimum. | https://www.tensortonic.com/problems/argmin |
| Evaluate a Scalar Neuron | Evaluate a scalar PyTorch tanh neuron from aligned inputs, weights, and bias using promoted floating-point types. | https://www.tensortonic.com/problems/autograd-l04-neuron-forward |
| Differentiate a Tanh Activation | Evaluate scalar tanh and manually combine its local derivative with an upstream gradient. | https://www.tensortonic.com/problems/autograd-l04-tanh-forward-backward |
| Bag-of-Words Vector | Build a NumPy bag-of-words count vector from an ordered vocabulary while ignoring out-of-vocabulary tokens. | https://www.tensortonic.com/problems/bag-of-words |
| Batch Shuffling & Mini-Batch Generator | Create shuffled mini-batches from NumPy feature and target arrays with reproducible ordering and final-batch handling. | https://www.tensortonic.com/problems/batch-generator |
| Batch Normalization (Forward) | Implement the batch-normalization forward pass in NumPy using feature-wise statistics, scale, shift, and numerical stability. | https://www.tensortonic.com/problems/batch-normalization |
| Bernoulli Probability Mass Function & Moments | Compute the Bernoulli probability mass function, expected value, and variance for a valid success probability. | https://www.tensortonic.com/problems/bernoulli-pmf |
| Binomial Probability Mass Function | Compute binomial probability mass and cumulative probabilities from trial count, success probability, and outcome. | https://www.tensortonic.com/problems/binomial-pmf-cdf |
| Bootstrap Mean & Confidence Interval | Estimate a sample mean and confidence interval through reproducible bootstrap resampling of numeric observations. | https://www.tensortonic.com/problems/bootstrap-mean |
| Implement Causal Masking for Attention | Create a causal attention mask that blocks each token from attending to future positions in a sequence. | https://www.tensortonic.com/problems/causal-masking |
| 1D Convolution | Implement 1D cross-correlation in CUDA with one thread per output element and register accumulation over the kernel. | https://www.tensortonic.com/problems/conv-1d |
| 2D Convolution | Implement 2D cross-correlation in CUDA with a two-dimensional grid, row-major indexing, and per-output accumulation. | https://www.tensortonic.com/problems/conv-2d |
| 3D Convolution | Implement naive 3D cross-correlation in CUDA with a three-dimensional launch grid and row-major flattened indexing. | https://www.tensortonic.com/problems/conv-3d |
| Cosine Annealing LR Scheduler | Compute a cosine-annealed learning rate between configured maximum and minimum values across training steps. | https://www.tensortonic.com/problems/cosine-annealing-lr |
| Implement Cosine Similarity | Implement cosine similarity from scratch with NumPy. Covers dot products, Euclidean norms and zero-vector handling, with tests in the browser. | https://www.tensortonic.com/problems/cosine-similarity |
| Compute Covariance Matrix | Compute a sample covariance matrix from centered observations, preserving feature-to-feature relationships. | https://www.tensortonic.com/problems/covariance-matrix |
| Cross-Entropy Loss | Implement numerically stable mean cross-entropy loss in CUDA using row-wise log-sum-exp and batch reduction. | https://www.tensortonic.com/problems/cross-entropy |
| Implement Cross-Entropy Loss | Compute multiclass cross-entropy loss from class probabilities and integer labels with stable logarithms. | https://www.tensortonic.com/problems/cross-entropy-loss |
| Cosine Similarity | Compute cosine similarity in CUDA by reducing the dot product and both vector norms into one scalar result. | https://www.tensortonic.com/problems/cuda-cosine-similarity |
| Dot Product | Implement a multi-block CUDA dot-product reduction that combines partial sums into one scalar output. | https://www.tensortonic.com/problems/cuda-dot-product |
| GELU | Implement exact GELU activation in CUDA with one thread per element and the device error-function intrinsic. | https://www.tensortonic.com/problems/cuda-gelu |
| Leaky ReLU | Implement Leaky ReLU activation in CUDA with one thread per element, bounds checks, and a configurable negative slope. | https://www.tensortonic.com/problems/cuda-leaky-relu |
| Matrix Transpose | Implement matrix transpose in CUDA with a two-dimensional launch grid, row-major buffers, and bounds-checked writes. | https://www.tensortonic.com/problems/cuda-matrix-transpose |
| Reparameterized Gaussian Sample | Draw Gaussian samples with the reparameterization trick, writing each sample as a deterministic function of noise. | https://www.tensortonic.com/problems/diffusion-l01-reparameterized-gaussian-sample |
| Implement Dot Product | Implement the dot product of equal-length numeric vectors by summing element-wise products without library shortcuts. | https://www.tensortonic.com/problems/dot-product |
| Dropout | Implement inverted dropout in CUDA using a supplied binary mask and scaling retained elements by the keep probability. | https://www.tensortonic.com/problems/dropout |
| Calculate Eigenvalues of a Matrix | Calculate the eigenvalues of a square matrix and return them in the format required by the numerical contract. | https://www.tensortonic.com/problems/eigenvalues |
| ELU Activation | Apply the ELU activation element-wise, retaining positive inputs and exponentially transforming negative values. | https://www.tensortonic.com/problems/elu-activation |
| Compute Entropy for a Node | Compute decision-tree node entropy from class labels using empirical class probabilities and base-two logarithms. | https://www.tensortonic.com/problems/entropy-node |
| Implement Euclidean Distance | Compute Euclidean distance between equal-length NumPy vectors as the square root of summed squared differences. | https://www.tensortonic.com/problems/euclidean-distance |
| Expected Value (Discrete Distribution) | Compute the expected value of a discrete distribution from matched outcomes and normalized probabilities. | https://www.tensortonic.com/problems/expected-value-discrete |
| Implement GELU Activation (Gaussian Error Linear Unit) | Implement the exact Gaussian Error Linear Unit activation element-wise with the erf function, with the formula and tests in the browser. | https://www.tensortonic.com/problems/gelu |
| Matrix-Vector Multiplication | Implement row-major CUDA matrix-vector multiplication with one thread computing each output row. | https://www.tensortonic.com/problems/gemv |
| Geometric Probability Mass Function & Mean | Compute the geometric distribution probability mass and mean from a valid success probability. | https://www.tensortonic.com/problems/geometric-pmf-mean |
| Compute Gini Impurity for a Split | Compute weighted Gini impurity for a candidate decision-tree split from the class labels on both sides. | https://www.tensortonic.com/problems/gini-impurity |
| Gradient Clipping (Global Norm) | Clip a NumPy gradient array by its global L2 norm while preserving direction when scaling is required. | https://www.tensortonic.com/problems/gradient-clipping |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| Hadamard Product | Implement elementwise matrix multiplication in CUDA using a two-dimensional grid and row-major bounds-checked indexing. | https://www.tensortonic.com/problems/hadamard-product |
| He Initialization | Implement He weight initialization from scratch in NumPy. See the formula, why fan-in scaling matters, and test your code in the browser. | https://www.tensortonic.com/problems/he-initialization |
| Implement Hinge Loss (Binary SVM) | Compute binary SVM hinge loss from signed labels and prediction scores using the required margin. | https://www.tensortonic.com/problems/hinge-loss |
| Impute Missing Values (mean/median) | Impute missing numeric values column-wise with either the mean or median while leaving observed values unchanged. | https://www.tensortonic.com/problems/impute-missing |
| Compute Information Gain for a Split | Compute information gain for a decision-tree split from parent entropy and weighted child entropies. | https://www.tensortonic.com/problems/information-gain |
| K-Means Assignment Step | Assign each sample to its nearest K-means centroid using Euclidean distance and deterministic tie handling. | https://www.tensortonic.com/problems/k-means-assignment |
| K-Means Centroid Update | Update K-means centroids as cluster means while applying the required behavior for empty clusters. | https://www.tensortonic.com/problems/k-means-centroid-update |
| K-Fold Split (Indices Only) | Generate deterministic K-fold train and validation index splits that use every sample exactly once for validation. | https://www.tensortonic.com/problems/kfold-split |
| Implement KL Divergence | Compute Kullback-Leibler divergence between discrete distributions in NumPy, with safe handling of zero probabilities and tests in the browser. | https://www.tensortonic.com/problems/kl-divergence |
| KNN Distance + Neighbor Lookup | Find the nearest neighbors of a query point by computing and ordering Euclidean distances to training samples. | https://www.tensortonic.com/problems/knn-distance |
| L1 Normalization | Implement CUDA L1 vector normalization by reducing absolute values and dividing each element by the resulting norm. | https://www.tensortonic.com/problems/l1-normalize |
| L2 Normalization | Implement CUDA L2 vector normalization by reducing squared values and dividing each element by the resulting norm. | https://www.tensortonic.com/problems/l2-normalize |
| Layer Normalization | Implement fused row-wise LayerNorm in CUDA with shared-memory mean and variance reduction, affine scale, and bias. | https://www.tensortonic.com/problems/layer-norm |
| Implement Leaky ReLU (with α) | Apply Leaky ReLU element-wise with a configurable negative slope while retaining positive inputs. | https://www.tensortonic.com/problems/leaky-relu |
| Linear Regression Closed Form | Fit linear regression with the closed-form normal equation and return coefficients for the supplied design matrix. | https://www.tensortonic.com/problems/linear-regression-closed-form |
| Logistic Regression Training Loop | Train binary logistic regression in NumPy using sigmoid probabilities, gradient descent, and learned weight and bias parameters. | https://www.tensortonic.com/problems/logistic-regression-training |
| Implement Majority Class Classifier | Fit a majority-class baseline and predict the most frequent training label for every requested sample. | https://www.tensortonic.com/problems/majority-classifier |
| Make Diagonal Matrix | Construct a square diagonal matrix from a one-dimensional vector while setting every off-diagonal entry to zero. | https://www.tensortonic.com/problems/make-diagonal |
| Implement Manhattan Distance | Compute Manhattan distance between equal-length vectors by summing absolute coordinate differences. | https://www.tensortonic.com/problems/manhattan-distance |
| Matrix Addition | Implement elementwise matrix addition in CUDA with a two-dimensional grid, row-major indexing, and bounds checks. | https://www.tensortonic.com/problems/matrix-addition |
| Matrix Inverse | Compute a square matrix inverse in NumPy while returning no result for invalid, non-square, or singular inputs. | https://www.tensortonic.com/problems/matrix-inverse |
| Matrix Multiplication | Implement row-major matrix multiplication in CUDA with one thread per output element and inner-product accumulation. | https://www.tensortonic.com/problems/matrix-multiplication |
| Implement Matrix Normalization | Normalize a NumPy matrix using the specified axis and norm while safely handling zero-magnitude slices. | https://www.tensortonic.com/problems/matrix-normalization |
| Matrix Trace | Compute the trace of a square matrix by summing its main diagonal entries without changing the input. | https://www.tensortonic.com/problems/matrix-trace |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Max of Array | Implement a multi-block CUDA maximum reduction that combines block-local maxima into one scalar output. | https://www.tensortonic.com/problems/max-of-array |
| Max Pool 2D | Implement 2D max pooling in CUDA with one thread per output, configurable windows and strides, and row-major indexing. | https://www.tensortonic.com/problems/max-pool-2d |
| Mean, Median, Mode | Calculate the mean, median, and deterministic mode of a numeric collection, including tied frequencies. | https://www.tensortonic.com/problems/mean-median-mode |
| Mean Squared Error (MSE) | Compute mean squared error between predictions and targets by averaging their squared element-wise differences. | https://www.tensortonic.com/problems/mean-squared-error |
| Mean and Variance | Compute population mean and variance in CUDA with device-side reductions and synchronized scalar outputs. | https://www.tensortonic.com/problems/mean-variance |
| Min of Array | Implement a multi-block CUDA minimum reduction that combines block-local minima into one scalar output. | https://www.tensortonic.com/problems/min-of-array |
| Implement Nadam (Nesterov + Adam) | Implement one Nadam optimizer step in NumPy by combining Adam moments with Nesterov momentum. | https://www.tensortonic.com/problems/nadam-optimizer |
| Implement Nesterov Momentum (NAG) | Implement a Nesterov accelerated-gradient update using lookahead momentum and the current gradient. | https://www.tensortonic.com/problems/nesterov-momentum |
| One-Hot Encoding (Multi-class) | Convert multiclass integer labels into a NumPy one-hot matrix with one active column per sample. | https://www.tensortonic.com/problems/one-hot-encoding |
| Outer Product | Compute a vector outer product in CUDA with a two-dimensional grid, row-major output, and bounds-checked indexing. | https://www.tensortonic.com/problems/outer-product |
| Pad Sequences | Pad or truncate variable-length token ID sequences in NumPy with configurable maximum length and padding values. | https://www.tensortonic.com/problems/pad-sequences |
| PCA Projection | Project centered observations onto supplied principal components to produce lower-dimensional features. | https://www.tensortonic.com/problems/pca-projection |
| Compute Pearson Correlation Matrix | Compute the Pearson correlation matrix between numeric features using centered covariance and standard deviations. | https://www.tensortonic.com/problems/pearson-correlation |
| Percentiles / Quantiles | Calculate requested percentiles from numeric data using the interpolation rule specified by the problem. | https://www.tensortonic.com/problems/percentiles |
| Perplexity Computation | Compute language-model perplexity from token probability distributions and the observed token indices. | https://www.tensortonic.com/problems/perplexity-computation |
| Poisson Probability Mass Function & Cumulative Distribution Function | Compute Poisson probability mass and cumulative probabilities for a nonnegative event count and rate. | https://www.tensortonic.com/problems/poisson-pmf-cdf |
| Implement Positional Encoding (sin/cos) | Generate sinusoidal Transformer positional encodings across sequence positions and embedding dimensions, with the formula and tests in the browser. | https://www.tensortonic.com/problems/positional-encoding |
| Precision and Recall at K | Compute precision@K and recall@K for recommendation systems by comparing ranked predictions with relevant items, with tests in the browser. | https://www.tensortonic.com/problems/precision-recall-at-k |
| Random Forest Majority Vote | Combine multiple decision-tree predictions with majority voting and deterministic handling of tied classes. | https://www.tensortonic.com/problems/random-forest-vote |
| ReLU | Implement ReLU activation in CUDA with one thread per element, bounds checks, and branch-efficient rectification. | https://www.tensortonic.com/problems/relu |
| Implement ReLU Activation | Apply the ReLU activation element-wise by replacing negative values with zero and preserving nonnegative inputs. | https://www.tensortonic.com/problems/relu-activation |
| Remove Stopwords | Remove tokens found in a supplied stopword collection while preserving the order of remaining words. | https://www.tensortonic.com/problems/remove-stopwords |
| Ridge Regression | Fit ridge regression with L2 regularization using the closed-form solution required by the problem. | https://www.tensortonic.com/problems/ridge-regression |
| RMS Normalization | Implement row-wise RMS normalization in CUDA with sum-of-squares reduction, numerical stability, and learnable scaling. | https://www.tensortonic.com/problems/rms-norm |
| RMSProp Optimizer (Single Update Step) | Implement one RMSProp update in NumPy using an exponential squared-gradient average and adaptive scaling. | https://www.tensortonic.com/problems/rmsprop-optimizer |
| RNN Step Forward (Tanh Cell) | Implement one vanilla RNN timestep with affine input and recurrent transforms followed by tanh activation. | https://www.tensortonic.com/problems/rnn-step-forward |
| Sample Variance & Standard Deviation | Compute sample variance and standard deviation with Bessel's correction from a numeric collection. | https://www.tensortonic.com/problems/sample-var-std |
| Scaled Dot-Product Attention | Implement single-head scaled dot-product attention in CUDA with stable row-wise softmax and value aggregation. | https://www.tensortonic.com/problems/scaled-dot-product-attention |
| SELU Activation | Apply SELU activation element-wise with scaled positive values and exponential negative values. | https://www.tensortonic.com/problems/selu-activation |
| Sigmoid | Implement sigmoid activation in CUDA with one thread per element, device exponential math, and bounds-checked memory access. | https://www.tensortonic.com/problems/sigmoid |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Softmax | Implement numerically stable CUDA softmax over a vector using global maximum and normalization reductions. | https://www.tensortonic.com/problems/softmax |
| Implement Softmax Function | Implement numerically stable softmax by shifting logits before exponentiation and normalizing probabilities. | https://www.tensortonic.com/problems/softmax-function |
| Sum of Array | Implement a multi-block CUDA sum reduction that combines partial block sums into one scalar output. | https://www.tensortonic.com/problems/sum-of-array |
| Swish | Implement fused Swish or SiLU activation in CUDA with one thread per element and device exponential math. | https://www.tensortonic.com/problems/swish |
| Implement Swish Activation | Apply the Swish activation element-wise in NumPy by multiplying each input by its sigmoid, computed without overflow. | https://www.tensortonic.com/problems/swish-activation |
| One-Sample t-Test | Compute a one-sample t-statistic in NumPy using the sample mean, Bessel-corrected deviation, and hypothesized mean. | https://www.tensortonic.com/problems/t-test-one-sample |
| Tanh | Implement hyperbolic tangent activation in CUDA with one thread per element, device intrinsic math, and bounds checks. | https://www.tensortonic.com/problems/tanh |
| Implement Tanh Activation | Implement the hyperbolic tangent activation element-wise with outputs bounded between minus one and one. | https://www.tensortonic.com/problems/tanh-activation |
| Target Encoding | Encode each categorical value with the mean target observed for its category while preserving row order. | https://www.tensortonic.com/problems/target-encoding |
| Text Chunking | Split text into ordered chunks under the requested size and overlap rules without dropping content. | https://www.tensortonic.com/problems/text-chunking |
| Tiled Matrix Multiplication | Implement shared-memory tiled matrix multiplication in CUDA with synchronized tile loads and partial-tile handling. | https://www.tensortonic.com/problems/tiled-matmul |
| Fused Multiply-Add | Implement a Triton fused multiply-add kernel with contiguous tiles, hardware FMA, and masked tail handling. | https://www.tensortonic.com/problems/triton-fused-multiply-add |
| Fused Row-Wise Softmax | Implement fused row-wise softmax in Triton with stable max subtraction, register reductions, and masked column tails. | https://www.tensortonic.com/problems/triton-fused-softmax |
| GELU | Implement exact GELU activation in Triton with device error-function math and masked contiguous tiles. | https://www.tensortonic.com/problems/triton-gelu |
| L2 Vector Norm | Compute a Triton L2 vector norm with tiled sum-of-squares reduction, atomic accumulation, and masked tail lanes. | https://www.tensortonic.com/problems/triton-l2-norm |
| Row-Wise LogSumExp | Implement numerically stable row-wise LogSumExp in Triton with max subtraction and masked register reductions. | https://www.tensortonic.com/problems/triton-logsumexp |
| Vector Max Reduction | Compute a vector maximum with one Triton reduction program and masked tail lanes that cannot win comparisons. | https://www.tensortonic.com/problems/triton-max |
| Single-Pass Mean and Variance | Compute population mean and variance in Triton with single-pass statistics, atomic partials, and masked tails. | https://www.tensortonic.com/problems/triton-mean-variance |
| ReLU | Implement ReLU activation in Triton with contiguous program tiles, branch-free rectification, and masked tails. | https://www.tensortonic.com/problems/triton-relu |
| SiLU | Implement fused SiLU or Swish activation in Triton with contiguous tiles, sigmoid weighting, and masked tails. | https://www.tensortonic.com/problems/triton-silu |
| Vector Sum Reduction | Implement tiled vector sum reduction in Triton with register partials, atomic accumulation, and masked tail lanes. | https://www.tensortonic.com/problems/triton-sum |
| Vector Addition | Implement elementwise vector addition in Triton with contiguous program tiles and safe masking for partial tails. | https://www.tensortonic.com/problems/triton-vector-addition |
| Vector Addition | Implement bounds-checked pointwise vector addition in CUDA with one thread per output element. | https://www.tensortonic.com/problems/vector-addition |
| Vector Subtraction | Implement bounds-checked pointwise vector subtraction in CUDA with one thread per output element. | https://www.tensortonic.com/problems/vector-subtract |
| Warmup + Linear Decay LR Schedule | Compute a learning-rate schedule with linear warmup followed by linear decay across training steps. | https://www.tensortonic.com/problems/warmup-decay-lr |
| Xavier Initialization | Scale raw weights into the Xavier uniform range using a bound derived from fan-in and fan-out. | https://www.tensortonic.com/problems/xavier-initialization |
| Implement z-Score Standardization | Standardize NumPy features to zero mean and unit variance with explicit handling for constant columns. | https://www.tensortonic.com/problems/zscore-standardization |
| Fine-tuning Architecture | Build BERT fine-tuning utilities for freezing encoder layers and producing sequence or token classification logits. | https://www.tensortonic.com/research/bert/bert-fine-tuning |
| Masked Language Modeling | Implement BERT masked language modeling with the 80-10-10 replacement strategy, training labels, and vocabulary logits. | https://www.tensortonic.com/research/bert/bert-masked-lm |
| Next Sentence Prediction | Create BERT next-sentence prediction pairs and compute binary classification logits for IsNext and NotNext examples. | https://www.tensortonic.com/research/bert/bert-nsp |
| BERT Pooler | Implement the BERT pooler by projecting the first token's hidden state through a dense layer and tanh activation. | https://www.tensortonic.com/research/bert/bert-pooler |
| Segment Embeddings | Build BERT input embeddings by summing learned token, position, and sentence-segment embedding vectors. | https://www.tensortonic.com/research/bert/bert-segment-embedding |
| WordPiece Tokenization | Implement BERT WordPiece tokenization with greedy longest-match subwords, continuation prefixes, and unknown-token fallback. | https://www.tensortonic.com/research/bert/bert-wordpiece |
| Forward Diffusion Process | Implement the DDPM forward diffusion process by mixing clean samples with Gaussian noise at a selected timestep. | https://www.tensortonic.com/research/ddpm/ddpm-forward |
| DDPM Training Loss | Compute the DDPM training objective as mean squared error between sampled noise and the model's noise prediction. | https://www.tensortonic.com/research/ddpm/ddpm-loss |
| Noise Schedule | Generate a DDPM noise schedule with beta values and cumulative alpha products used across diffusion timesteps. | https://www.tensortonic.com/research/ddpm/ddpm-schedule |
| GAN Discriminator | Implement a GAN discriminator that maps input samples through dense layers to real-versus-fake probabilities. | https://www.tensortonic.com/research/gan/gan-discriminator |
| Complete GAN System | Assemble a complete GAN system that generates samples, scores them with the discriminator, and returns both outputs. | https://www.tensortonic.com/research/gan/gan-full-network |
| GAN Generator | Implement a GAN generator that transforms latent noise through learned dense layers into generated samples. | https://www.tensortonic.com/research/gan/gan-generator |
| GAN Loss Functions | Compute numerically stable binary cross-entropy losses for the GAN generator and discriminator objectives. | https://www.tensortonic.com/research/gan/gan-loss |
| Mode Collapse Detection | Detect GAN mode collapse by measuring diversity across generated samples and flagging low-variance outputs. | https://www.tensortonic.com/research/gan/gan-mode-collapse |
| GAN Training Loop | Implement one GAN training iteration with separate discriminator and generator forward and update steps. | https://www.tensortonic.com/research/gan/gan-training-loop |
| Reset Gate | Implement a GRU reset gate that controls how much of the previous hidden state contributes to the candidate state. | https://www.tensortonic.com/research/gru/gru-reset-gate |
| Complete LSTM Cell | Build a complete LSTM cell with forget, input, candidate, cell-state, output, and hidden-state calculations. | https://www.tensortonic.com/research/lstm/lstm-cell |
| Cell State Update | Implement the LSTM cell-state update by combining retained memory with input-gated candidate information. | https://www.tensortonic.com/research/lstm/lstm-cell-state |
| Forget Gate | Implement an LSTM forget gate by combining the previous hidden state and current input with a sigmoid projection. | https://www.tensortonic.com/research/lstm/lstm-forget-gate |
| Input Gate | Implement the LSTM input gate and candidate activation that control new information written to the cell state. | https://www.tensortonic.com/research/lstm/lstm-input-gate |
| Output Gate | Implement the LSTM output gate and expose the current hidden state from the updated cell memory. | https://www.tensortonic.com/research/lstm/lstm-output-gate |
| BatchNorm in ResNet | Implement ResNet batch normalization with channel statistics, learned scale and bias, and training or inference behavior. | https://www.tensortonic.com/research/resnet/resnet-batch-norm |
| Bottleneck Block | Build a ResNet bottleneck block using 1x1 channel reduction, 3x3 convolution, and 1x1 channel expansion. | https://www.tensortonic.com/research/resnet/resnet-bottleneck |
| Convolutional Block | Implement a ResNet convolutional block with a projected shortcut that matches changed spatial and channel dimensions. | https://www.tensortonic.com/research/resnet/resnet-conv-block |
| Full ResNet Assembly | Assemble a ResNet forward pass from the stem, residual stages, global average pooling, and the classification head. | https://www.tensortonic.com/research/resnet/resnet-full-network |
| Identity Block | Implement a ResNet identity block with a three-layer bottleneck branch, batch normalization, ReLU, and an unchanged skip path. | https://www.tensortonic.com/research/resnet/resnet-identity-block |
| Skip Connection Analysis | Analyze ResNet skip connections by combining residual and identity tensors and tracking gradient flow through the addition. | https://www.tensortonic.com/research/resnet/resnet-skip-connection |
| RNN Cell | Implement an Elman RNN cell that combines the current input and previous hidden state before applying tanh. | https://www.tensortonic.com/research/rnn/rnn-cell |
| Hidden State | Initialize a vanilla RNN hidden state as a floating-point zero matrix for the requested batch and hidden dimensions. | https://www.tensortonic.com/research/rnn/rnn-hidden-state |
| Vanishing Gradients | Simulate vanishing or exploding RNN gradients by repeatedly applying the hidden matrix's spectral norm. | https://www.tensortonic.com/research/rnn/rnn-vanishing-gradients |
| Scaled Dot-Product Attention | Implement scaled dot-product attention in PyTorch using query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/transformer/transformers-attention |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Encoder Block | Assemble a Transformer encoder block with multi-head attention, residual paths, layer normalization, and a feed-forward network. | https://www.tensortonic.com/research/transformer/transformers-encoder-block |
| Feed-Forward Network | Implement the Transformer's position-wise feed-forward network with two linear projections and a ReLU activation. | https://www.tensortonic.com/research/transformer/transformers-feed-forward |
| Layer Normalization | Implement Transformer layer normalization in NumPy using per-token mean, variance, scale, and bias. | https://www.tensortonic.com/research/transformer/transformers-layer-normalization |
| Multi-Head Attention | Build NumPy multi-head attention with learned projections, per-head scaled attention, concatenation, and output projection. | https://www.tensortonic.com/research/transformer/transformers-multi-head-attention |
| Positional Encoding | Implement sinusoidal Transformer positional encodings in NumPy with alternating sine and cosine dimensions. | https://www.tensortonic.com/research/transformer/transformers-positional-encoding |
| Tokenization | Build a word-level Transformer tokenizer with fixed special-token IDs, sorted vocabulary entries, encoding, and decoding. | https://www.tensortonic.com/research/transformer/transformers-tokenization |
| VAE Decoder | Build a variational autoencoder decoder that maps sampled latent vectors back to reconstructed input probabilities. | https://www.tensortonic.com/research/vae/vae-decoder |
| ELBO Loss Function | Implement the VAE evidence lower bound from reconstruction loss and KL divergence with configurable weighting. | https://www.tensortonic.com/research/vae/vae-elbo-loss |
| VAE Encoder | Implement a variational autoencoder encoder that maps inputs to separate latent mean and log-variance vectors. | https://www.tensortonic.com/research/vae/vae-encoder |
| Complete VAE Network | Assemble a complete variational autoencoder forward pass with encoding, reparameterized sampling, and decoding. | https://www.tensortonic.com/research/vae/vae-full-network |
| KL Divergence Loss | Compute the VAE KL-divergence term between a diagonal Gaussian posterior and the standard normal prior. | https://www.tensortonic.com/research/vae/vae-kl-divergence |
| Reparameterization Trick | Implement the VAE reparameterization trick by sampling latent vectors from the predicted mean and log variance. | https://www.tensortonic.com/research/vae/vae-reparameterization |
| VGG Classifier Head | Build the VGG classifier by flattening spatial features and applying two ReLU hidden layers plus a logits projection. | https://www.tensortonic.com/research/vgg/vgg-classifier |
| VGG Configuration | Generate the canonical convolution and max-pooling layer configuration for VGG11, VGG13, VGG16, or VGG19. | https://www.tensortonic.com/research/vgg/vgg-config |
| Class Token [CLS] | Prepend a learned classification token to each Vision Transformer patch sequence for image-level prediction. | https://www.tensortonic.com/research/vit/vit-class-token |
| ViT Encoder Block | Build a Vision Transformer encoder block with layer normalization, multi-head attention, MLP, and residual connections. | https://www.tensortonic.com/research/vit/vit-encoder-block |
| Complete Vision Transformer | Assemble a complete Vision Transformer from patch embedding, class and position tokens, encoder blocks, and classifier. | https://www.tensortonic.com/research/vit/vit-full-network |
| Classification Head | Implement the Vision Transformer classification head by normalizing and projecting the final class-token representation. | https://www.tensortonic.com/research/vit/vit-mlp-head |
| Patch Embedding | Implement Vision Transformer patch embeddings by splitting images into fixed patches and linearly projecting each patch. | https://www.tensortonic.com/research/vit/vit-patch-embedding |
| Position Embedding | Add learned positional embeddings to Vision Transformer patch-token sequences while preserving batch dimensions. | https://www.tensortonic.com/research/vit/vit-position-embedding |
| CBOW Forward Pass | Implement the Word2Vec CBOW forward pass by averaging context embeddings and producing vocabulary logits. | https://www.tensortonic.com/research/word2vec/word2vec-cbow-forward |
| Negative Sampling Distribution | Build the Word2Vec negative-sampling distribution from unigram counts raised to the three-quarter power. | https://www.tensortonic.com/research/word2vec/word2vec-noise-dist |
| Skip-gram Negative Sampling Loss | Implement skip-gram negative-sampling loss from center, positive-context, and negative-word embedding scores. | https://www.tensortonic.com/research/word2vec/word2vec-sgns-loss |
| Skip-gram Pair Generation | Generate Word2Vec skip-gram training pairs by pairing each center token with words inside its context window. | https://www.tensortonic.com/research/word2vec/word2vec-skipgram-pairs |
| Frequent-Word Subsampling | Implement Word2Vec frequent-word subsampling by computing token retention probabilities from corpus frequencies. | https://www.tensortonic.com/research/word2vec/word2vec-subsampling |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/dodi488)
<!-- tensortonic:end -->
