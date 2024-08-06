#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>

class Model {
public:
  Model(const std::string& model_path);
  ~Model();

  // Load the model from a file
  void load();

  // Run the model on input data
  std::vector<float> run(const std::vector<float>& input);

  // Get the model's input shape
  std::vector<int> get_input_shape() const;

  // Get the model's output shape
  std::vector<int> get_output_shape() const;

private:
  std::string model_path_;
  tensorflow::Session* session_;
  tensorflow::GraphDef graph_def_;
  std::vector<tensorflow::Tensor> inputs_;
  std::vector<tensorflow::Tensor> outputs_;
};

#endif  // MODEL_H
