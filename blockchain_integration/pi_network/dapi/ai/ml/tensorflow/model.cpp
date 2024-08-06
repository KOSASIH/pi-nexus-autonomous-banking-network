#include "model.h"
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>

Model::Model(const std::string& model_path) : model_path_(model_path) {}

Model::~Model() {
  if (session_) {
    delete session_;
  }
}

void Model::load() {
  // Load the model from a file
  tensorflow::NewSession(tensorflow::SessionOptions(), &session_);
  tensorflow::LoadSavedModel(session_, {tensorflow::kSavedModelTagServe}, model_path_, &graph_def_);
  tensorflow::GraphDef graph_def;
  TF_CHECK_OK(tensorflow::NewSession(tensorflow::SessionOptions(), &session_));
  TF_CHECK_OK(session_->Create(graph_def));
  TF_CHECK_OK(session_->Run({}, {}, {"input"}, &inputs_, &outputs_));
}

std::vector<float> Model::run(const std::vector<float>& input) {
  // Create a tensor for the input data
  tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, input.size()}));
  auto input_tensor_mapped = input_tensor.tensor<float, 2>();
  for (int i = 0; i < input.size(); i++) {
    input_tensor_mapped(0, i) = input[i];
  }

  // Run the model on the input data
  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session_->Run({{inputs_[0], input_tensor}}, {"output"}, {}, &outputs));

  // Extract the output data
  std::vector<float> output;
  auto output_tensor_mapped = outputs[0].tensor<float, 2>();
  for (int i = 0; i < output_tensor_mapped.dimension(1); i++) {
    output.push_back(output_tensor_mapped(0, i));
  }

  return output;
}

std::vector<int> Model::get_input_shape() const {
  return {1, inputs_[0].shape().dim_size(1)};
}

std::vector<int> Model::get_output_shape() const {
  return {1, outputs_[0].shape().dim_size(1)};
}
