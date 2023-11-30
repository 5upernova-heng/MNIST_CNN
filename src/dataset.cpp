#include "dataset.h"

namespace {
    constexpr int kTrainSize = 60000;
    constexpr int kTestSize = 10000;

    constexpr int kRows = 28;
    constexpr int kCols = 28;

    std::pair<torch::Tensor, torch::Tensor> StringToTensor(const std::string& line) {
        std::stringstream lineStream(line);
        std::vector<int> result;
        std::string cell;
        std::string label;
        std::getline(lineStream, label, ',');
        for (int i = 0; i < kRows * kCols; i++) {
            std::getline(lineStream, cell, ',');
            result.push_back(stoi(cell));
        }
        auto opts = torch::TensorOptions().dtype(torch::kInt32);
        torch::Tensor data = torch::from_blob(result.data(), { 1, 28, 28 }, torch::kInt32).to(torch::kInt64);
        return {data, torch::tensor(stoi(label), torch::kUInt8)};
    }
}

std::pair<torch::Tensor, torch::Tensor> read_data(const std::string& root, bool  train) {
    std::string filename = train ? "mnist_train.csv" : "mnist_test.csv";
    std::ifstream ifs(root + "/" + filename);
    std::string line;
    auto num_samples = train ? kTrainSize : kTestSize;
    auto images = torch::empty({num_samples, 1, kRows, kCols}, torch::kUInt8);
    auto labels = torch::empty(num_samples, torch::kUInt8);
    int index = 0;
    std::getline(ifs, line); // first line (columns) is useless here.
    while (std::getline(ifs, line)) {
        auto line_data = StringToTensor(line);
        images[index] = std::move(line_data.first);
        labels[index] = std::move(line_data.second);
        index++;
    }
    return {images, labels};
}

DigitDataset::DigitDataset(const std::string &root, DigitDataset::Mode mode) : mode_(mode) {
    auto data = read_data(root, mode == Mode::kTrain);

    images_ = std::move(data.first);
    targets_ = std::move(data.second);
}

torch::data::Example<> DigitDataset::get(size_t index) {
    return {images_[index], targets_[index]};
}

bool DigitDataset::is_train() const noexcept {
    return mode_ == Mode::kTrain;
}

torch::optional<size_t> DigitDataset::size() const {
    return images_.size(0);
}

const torch::Tensor& DigitDataset::images() const {
    return images_;
}

const torch::Tensor& DigitDataset::targets() const {
    return targets_;
}
