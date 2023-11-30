#include <iostream>
#include <fstream>
#include <torch/torch.h>

struct DigitDataset : torch::data::datasets::Dataset<DigitDataset> {
public:
	enum Mode {kTrain, kTest};
	explicit DigitDataset(const std::string& root, Mode mode = Mode::kTrain);

	torch::data::Example<> get(size_t index) override;
	torch::optional<size_t> size() const override;
	bool is_train() const noexcept;
	const torch::Tensor& images() const;
	const torch::Tensor& targets() const;

private:
	torch::Tensor images_;
	torch::Tensor targets_;
	Mode mode_;
};
