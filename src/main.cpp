#include <string>
#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include "dataset.h"
#include "network.h"

const int64_t kTrainBatchSize = 64;
const int64_t kTestBatchSize = 1000;
const int64_t kNumberOfEpochs = 10;

template <typename DataLoader>
void train(size_t epoch, Net& model, torch::Device device, DataLoader& data_loader, torch::optim::Optimizer& optimizer,
           size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

template <typename DataLoader>
void test(
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                     torch::Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
}


int main() {

    torch::manual_seed(1);

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "Training on CUDA" << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU" << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    Net model;
    model.to(device);


    std::string root = "../data/";
    auto train_dataset = DigitDataset(root)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());
    const size_t train_dataset_size = train_dataset.size().value();
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(train_dataset), kTrainBatchSize);

    auto test_dataset = DigitDataset(root, DigitDataset::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(test_dataset), kTestBatchSize);


    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

    for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
      train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
      test(model, device, *test_loader, test_dataset_size);
    }

}
