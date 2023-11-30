#include <torch/torch.h>

const int64_t kLogInterval = 10;

struct Net : torch::nn::Module {
  explicit Net();

  torch::Tensor forward(torch::Tensor x);

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::Dropout2d conv2_drop;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};


