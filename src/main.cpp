#include <iostream>
#include <string>
#include <torch/torch.h>
#include "dataset.h"

const int64_t kTrainBatchSize = 64;
const int64_t kTestBatchSize = 1000;

int main() {
    torch::manual_seed(1);

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA" << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "CPU" << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    std::string root = "../../data/";
    auto train_dataset = DigitDataset(root)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), kTrainBatchSize);
    auto test_dataset = DigitDataset(root, DigitDataset::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(test_dataset), kTestBatchSize);

}
