#include "model.h"

Model::Model(string model) {
    // load model
    if (!fs::exists(model)) {
        cerr << "model missing\n";
        return;
    }

    try {
        module = torch::jit::load(model);
    } catch (const c10::Error& e) {
        cerr << "error loading the model\n";
        cerr << e.what();
        cerr << e.msg();
      return;
    }
    cout << "Model load ok.\n";

    module.to(at::Device("cpu"));
}

torch::Tensor Model::PreprocessImage(cv::Mat img) {
    cv::resize(img, img, cv::Size(480,848));

    img.convertTo(img, CV_32FC3, 1/255.0);

    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, c10::kFloat);

    return img_tensor.clone();
}

void Model::Inference(cv::Mat frame) {
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(PreprocessImage(frame));

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    
    std::tuple<torch::Tensor, torch::Tensor> result = torch::max(output, 1);

    cout << "\n" << output[0][0].item<float>() << " steeringAngle \n";
    cout << output[0][1].item<int>() << " throttlePercentage \n";
    // cout << output[0][2].item<int>() << " brakePercentage \n";

    float steeringAngle = output[0][0].item<float>();
    int throttlePercentage = output[0][1].item<int>();
    // int brakePercentage = output[0][2].item<int>();

    // steering
    CommunicationStrategy::actuators.steeringAngle = LimitOutputFloat(steeringAngle);

    // throttle
    CommunicationStrategy::actuators.throttlePercentage = LimitOutputInt(throttlePercentage);
}

int Model::LimitOutputInt(int value, int min, int max) {
    if (value < min) {
        cerr << "error prediction is " << value << " but should have been above " << min << "\n";
        return min;
    } else if (value > max) {
        cerr << "error prediction is " << value << " but should have been below " << max << "\n";
        return max;
    } else {
        return value;
    }
}

float Model::LimitOutputFloat(float value, int min, int max) {
    if (value < min) {
        cerr << "error prediction is " << value << " but should have been above " << min << "\n";
        return min;
    } else if (value > max) {
        cerr << "error prediction is " << value << " but should have been below " << max << "\n";
        return max;
    } else {
        return value;
    }
}
