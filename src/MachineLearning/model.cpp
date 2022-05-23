#ifdef linux
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

void Model::EnableCSV() {
    if (!CSVIsEnabled) {
        csvFile.open(fs::current_path().string()+"/logs/ml/"+Time::currentDateTime()+"-images-ml.csv");
        csvFile << "Steer,Throttle,Brake,Image\n";
        CSVIsEnabled = true;
    }
}

void Model::closeCSV(){
    if (!csvIsClosed) {
        csvFile.close();
        csvIsClosed = true;
    }
}

torch::Tensor Model::PreprocessImage(cv::Mat img) {
    cv::resize(img, img, cv::Size(848,480));
    img.convertTo(img, CV_32FC3, 1/255.0);

    // crop image
    int y0 = 160;
    int y1 = 325;
    int x0 = 0;
    int x1 = 848;
    cv::Rect roi(/*x=*/x0, /*y=*/y0, /*w=*/x1-x0, /*h=*/y1-y0);
    cv::Mat crop = img(roi);

    torch::Tensor img_tensor = torch::from_blob(crop.data, {crop.rows, crop.cols, 3}, c10::kFloat);
    return img_tensor.clone();
}

void Model::Inference(cv::Mat frame, string img) {
    if (csvIsClosed) {
        return;
    }

    // Create a vector of inputs.
    vector<torch::jit::IValue> inputs;
    PreprocessImage(frame);
    inputs.push_back(PreprocessImage(frame));

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    // cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << endl;
    
    tuple<torch::Tensor, torch::Tensor> result = torch::max(output, 1);

    float steeringAngle = output[0][0].item<float>();
    int throttlePercentage = output[0][1].item<int>();
    // int brakePercentage = output[0][2].item<int>();

    cout << steeringAngle << " steeringAngle" << endl;
    cout << throttlePercentage << " throttlePercentage" << endl;
    // cout << brakePercentage << " brakePercentage" << endl;

    // steering
    steeringAngle = LimitOutputFloat(steeringAngle);
    CommunicationStrategy::actuators.steeringAngle = steeringAngle;

    // throttle
    throttlePercentage = LimitOutputInt(throttlePercentage);
    CommunicationStrategy::actuators.throttlePercentage = throttlePercentage;

    if (!csvIsClosed && img != "") {
        csvFile << std::to_string(steeringAngle)+","+std::to_string(throttlePercentage)+",0,"+img+"\n";
    }
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
#endif