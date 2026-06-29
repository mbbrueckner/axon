#include "axon/datasets/mnist.hpp"

#include <fstream>

namespace {
int32_t read_big_endian(std::ifstream& file) {
  uint8_t bytes[4];
  file.read(reinterpret_cast<char*>(bytes), 4);
  return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}
}  // namespace

namespace axon::datasets {

MNIST::MNIST(const std::string& images_path, const std::string& labels_path) {
  std::ifstream images_file(images_path, std::ios::binary);
  if (!images_file.is_open()) {
    throw std::runtime_error("Cannot open file: " + images_path);
  }

  // read header (4 x 32 bit) and check magic (first)
  if (int32_t magic = read_big_endian(images_file); magic != 2051) {
    throw std::invalid_argument(
        std::format("Magic number mismatch, expected 2051, got {}", magic));
  }

  int32_t n_images = read_big_endian(images_file);
  int32_t rows = read_big_endian(images_file);
  int32_t cols = read_big_endian(images_file);

  // read pixels
  std::vector<uint8_t> pixels(n_images * rows * cols);
  images_file.read(reinterpret_cast<char*>(pixels.data()), pixels.size());

  // normalize, convert to float-Vector
  std::vector<float> image_data(pixels.size());
  std::ranges::transform(
      pixels, image_data.begin(), [](uint8_t p) { return p / 255.0f; });

  // safe as tensor
  images_ = Tensor::from_data(image_data, {n_images, rows * cols});

  std::ifstream label_file(labels_path, std::ios::binary);
  if (!label_file.is_open()) {
    throw std::runtime_error("Cannot open file: " + labels_path);
  }

  // read header (1 x 32 bit) and check magic
  if (int32_t magic = read_big_endian(label_file); magic != 2049) {
    throw std::invalid_argument(
        std::format("Magic number mismatch, expected 2049, got {}", magic));
  }

  int32_t n_labels = read_big_endian(label_file);
  std::vector<uint8_t> labels(n_labels);
  label_file.read(reinterpret_cast<char*>(labels.data()), labels.size());
  
  labels_ = std::vector<idx_t>(labels.begin(), labels.end());
}

}  // namespace axon::datasets
