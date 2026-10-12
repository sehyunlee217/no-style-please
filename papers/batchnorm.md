---
layout: page
title: Batch Normalization
permalink: /papers/batchnorm.html
---

{% raw %}
- Paper: <https://arxiv.org/pdf/1502.03167>
- Github: <https://github.com/sehyunlee217/paper_implementations/tree/main/batch_norm>

### Paper Summary

- **Internal Covariate shifts**, defined as *change in the distribution of network activations due to the change in network parameters during training*, are an issue as after each layer, the input distributions can change and it can slow down training.
- Let us define a 2-layer network: $$L = F_{2}(F_{1}(u, \theta_{1}), \theta_{2})$$. For $$F_{2}$$, its inputs can be seen as $$x=F_{1}(u, \theta_{1})$$ then $$L = F_{2}(x, \theta_{2})$$. Therefore, if the distribution of the input $$x$$ changes at the layer, the parameters $$\theta_{2}$$ has to adjust according to the shift of $$x$$ at each layer.
- Saturating activations are another issue. For $$z = g(Wu + b)$$ with sigmoid $$g(x) = \frac{1}{1 + \exp(-x)}$$, large absolute inputs produce gradients close to zero.
- To demonstrate this, say we want to update the gradient with respect to the loss $$L$$. Then, due to the chain rule of partial derivatives, $$\frac{dL}{du} = \frac{dL}{dz} \times \frac{dz}{dx} \times
                    \frac{dx}{du} = \frac{dL}{dz} \times g^{'}(x) \times
                    W$$. But the $$g^{'}(x)$$ could likely be close to zero for larger absolute values, which suggests that (1) $$u$$ will receive very small signals to update (2) $$W$$ and $$b$$ will also update slowly due to the small gradients.
- BatchNorm helps keep the inputs in a more stable range, which can speed up training.

### Working towards Reducing Internal Covariate Shifts

- The gradient descent optimization needs to take into account that normalization takes place.
- For example, assume that we normalize the inputs ($$\hat{x} = x - E[x]$$) by subtracting the mean over the training data, where $$x = u + b$$.
- If the gradient is calculated while ignoring the normalization step; $$b \leftarrow b + \Delta b$$, where $$\frac{\delta L}{\delta b} = \frac{\delta L}{\delta \hat{x}}
                    \times \frac{\delta \hat{x}}{\delta x} \times \frac{\delta
                    x}{\delta b} = \frac{\delta L}{\delta \hat{x}} \times 1
                    \times 1 = \frac{\delta L}{\delta \hat{x}}$$. So $$\Delta b \propto - \frac{\delta L}{\delta
                    \hat{x}}$$, and we are ignoring the fact that the $$E[x]$$ term in $$\hat{x}$$ contains $$b$$.
- Then, $$x_{updated} = u + (b + \Delta b)$$ and $$E[x_{updated}] = E[u + (b + \Delta b)] = E[u + b] + \Delta
                    b$$.
- This results in $$\hat{x}_{updated} = x - E[x] = (u +b + \Delta b) - E[u +b]
                    + \Delta = u +b - E[u +b]$$
- In summary, $$\hat{x}_{updated} = u + b - E[u +b] = x - E[x]$$ suggesting that the bias $$b$$ was added to $$x$$ but the output of the layer $$\hat{x}$$ did not change as the bias is added then subtracted by the normalization part.
- If we go back to see $$\Delta b \propto - \frac{\delta L}{\delta
                    \hat{x}}$$, the optimizer will try to increase $$\Delta b$$ to decrease the loss but because $$\hat{x}$$ remains unchanged, $$\frac{\delta L}{\delta \hat{x}}$$ remains unchanged, so we can confirm that the loss remains unchanged as well.
- The paper mentions that this leads to $$b$$ growing indefinitely and blowing up.

**Key point**

- Instead, we need to include normalization when calculating gradients. Let us go back to the same example.
- Back to this step, here, we include that $$E[x]$$ contains $$b$$, then again $$\frac{\delta L}{\delta b} = \frac{\delta L}{\delta \hat{x}}
                    \times \frac{\delta \hat{x}}{\delta x} \times \frac{\delta
                    x}{\delta b}$$
- But this time taking into account $$E[x]$$ is a some function of $$x$$ or more specifically the batch mean that approximates $$x$$, $$\frac{1}{\delta x}(x - E[x]) = 1 - 1 = 0$$.
- Now, $$\frac{\delta L}{\delta b} = \frac{\delta L}{\delta \hat{x}}
                    \times 0 \times 1 = 0$$. So the optimizer does not have to adjust $$\Delta b$$ as it doesn't affect the loss.
- In this example, accounting for normalization prevents a pointless bias update that would be canceled out anyway.
- Furthermore, centering and scaling the inputs can help avoid the saturated regions of sigmoid. Normalization does not necessarily make their distribution Gaussian.

### Mini-Batch Normalization

- The paper normalizes the features independently instead of across all features by $$\hat{x}^{k} = \frac{x^{k} -
                    E[x^{k}]}{\sqrt{Var[x^{k}]}}​$$, where $$k$$ represents the independent features.
- Also note that because normalizing features can change the representations of the layers, the transformation has to be able to represent the identity transform (affine transform), which is accomplished through: $$y^{(k)} = \gamma^{k}\hat{x}^{k} + \beta^{k}$$.
- What it means to represent the identity transform is from $$y^{(k)} = \gamma^{k}\frac{x^{k} -
                    E[x^{k}]}{\sqrt{Var[x^{k}]}} + \beta^{k}$$, we can adjust both the parameters $$\gamma^{k}$$, $$\beta^{k}$$ to cancel out the $$Var[x^{k}]$$ part and the $$- E[x^{k}]$$ part such that $$y^{k} = x^{k}$$ in order to be able to recover the original activations before normalization if needed.
- The new parameters $$\gamma$$ and $$\beta$$ are learned and not canceled out because it is outside of the layer as well.
- During inference, we use $$\hat{x}$$ from the population rather than the mini batch as the normalization should be independent of the batch itself. In the case of the implementation, a moving average is used to track the population mean and variance.

#### Effects of Batch-Normalization

- Batch normalization can support higher learning rates and help avoid the saturated, nearly flat regions of sigmoid.
- Furthermore, batch norm is scale invariant for its parameters as $$BN((aW)u) = \frac{aWu - E[aWu]}{\sqrt{Var[aWu]}} =
                    \frac{a(Wu - E[Wu])}{a\sqrt{Var[Wu]}} = BN(Wu)$$.

### Implementation Details

- We add an $$\epsilon$$ for stability following the paper : $$\hat{x} = \frac{x_{i} -
                    \mu_{\beta}}{\sqrt{\sigma^{2}_{\beta} +
                    \epsilon}}​$$
- We calculate statistics over the batch and spatial dimensions for each channel. Initializing $$\gamma$$ to $$1$$ and $$\beta$$ to $$0$$ gives $$y^{(k)} = \hat{x}^{k}$$, so just the normalized input.
- The learnable affine parameters $$\gamma$$ and $$\beta$$ let the model adjust the normalized activations during backpropagation.
- Note the dimensions were matched during the normalization part by using `keepdim=True`

``` py
class BatchNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, x):
        # batchnorm across the number of features (channels) -> want to flatten other dims
        # x: (Batch, Channel, Width, Height) => (Batch, Channel, Feature_Size)
        init_shape = x.shape
        x = x.flatten(2)

        # take mean across features
        # mean, mean_sq : (Channel) -> keepdim=True -> (1, Channel, 1)
        mean = x.mean(dim=(0, 2), keepdim=True)
        mean_sq = (x**2).mean(dim=(0, 2), keepdim=True)

        # Var = E[x**2] - E[x]**2
        var = mean_sq - mean**2

        x_norm = (x - mean) / (torch.sqrt(var + self.eps))

        print(f"dim(x_norm) {x_norm.shape}")

        output = self.gamma * x_norm + self.beta

        return output.view(init_shape)
```

- For `x = torch.rand(32, 3, 64, 64)`, flattening the spatial dimensions gives `(32, 3, 64 * 64)`. Taking the mean over `dim=(0, 2)` gives `(1, 3, 1)`: one mean per channel, across the batch and spatial positions.
- Lastly we reshape it to the original `(32, 3, 64, 64)` as the input to the activation.

#### Training vs. Inference

- We noted above that during inference, we use the statistics on the population rather than the batches. The paper suggests a more efficient way, which is to take a running average of the mean and variances of the mini-batches during training.
- Here we use `register_buffer()` to store running statistics. They are part of the module state, but are not learnable parameters.
- We also introduce `momentum`, which is a hyperparameter used in PyTorch's `BatchNorm2d` with a default value of `0.1`, which we followed and will update the exponential moving averages for the running mean and variances.

``` py
class BatchNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5, train=True, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))

        self.train = train
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(1, num_channels, 1))
        self.register_buffer('running_variance', torch.ones(1, num_channels, 1))

    def forward(self, x):
        # batchnorm across the number of features (channels) -> want to flatten other dims
        # x: (Batch, Channel, Width, Height) => (Batch, Channel, Feature_Size)
        init_shape = x.shape
        x = x.flatten(2)

        if self.train:
            # take mean across features
            # mean, mean_sq : (Channel) -> keepdim=True -> (1, Channel, 1)
            mean = x.mean(dim=(0, 2), keepdim=True)
            mean_sq = (x**2).mean(dim=(0, 2), keepdim=True)

            # Var = E[x**2] - E[x]**2
            var = mean_sq - mean**2

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_variance = (1 - self.momentum) * self.running_variance + self.momentum * var
        
            mean = mean
            var = var
        else:
            mean = self.running_mean
            var = self.running_variance

        x_norm = (x - mean) / (torch.sqrt(var + self.eps))

        output = self.gamma * x_norm + self.beta

        return output.view(init_shape)
```
{% endraw %}
