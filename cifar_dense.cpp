#include <iostream>
#include <fstream>
#include <algorithm>
#include <cfloat>
#include <chrono>

// Function to read data from a text file into a pre-allocated array
void readDataFromFile(const std::string& filename, float* data, int& dataSize) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file - '" << filename << "'" << std::endl;
        return;
    }

    // Count the number of elements in the file to validate against dataSize
    float value;
    int count = 0;
    while (file >> value) {
        ++count;
    }

    if (count != dataSize) {
        std::cerr << "Data size mismatch. Expected " << dataSize << " elements, but file contains " << count << "." << std::endl;
        return;
    }

    // Go back to the beginning of the file
    file.clear();
    file.seekg(0, std::ios::beg);

    // Read the data into the array
    int index = 0;
    while (file >> value) {
        data[index++] = value;
    }

    file.close();
}

void maxpool2d(float* input_data, int input_channels, int input_height, int input_width,
               int pool_size, int stride, float* output_data) {
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;
    int total_iterations = input_channels * output_height * output_width;

    for (int index = 0; index < total_iterations; index++) {
        int c = index / (output_height * output_width);
        int h = (index / output_width) % output_height;
        int w = index % output_width;

        float max_val = -FLT_MAX;
        for (int p = 0; p < pool_size * pool_size; p++) {
            int ph = p / pool_size;
            int pw = p % pool_size;

            int input_h = h * stride + ph;
            int input_w = w * stride + pw;
            if (input_h < input_height && input_w < input_width) {
                int input_index = c * (input_height * input_width) + input_h * input_width + input_w;
                max_val = std::max(max_val, input_data[input_index]);
            }
        }
        int output_index = c * (output_height * output_width) + h * output_width + w;
        output_data[output_index] = max_val;
    }
}

void conv2d(float* input_data, int image_input_channels, int input_height, int input_width,
            float* weight_data, int weight_output_channels, int weight_input_channels, int weight_height, int weight_width,
            float* bias_data, int bias_number_of_elements, int kernel_size, int stride, int padding, 
            bool relu, float* output_data) {
    // Compute output dimensions
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    // Perform convolution with merged outer loops
    int total_iterations = weight_output_channels * output_height * output_width;
    for (int index = 0; index < total_iterations; ++index) {
        int out_channel = index / (output_height * output_width);
        int y = (index / output_width) % output_height;
        int x = index % output_width;

        float sum = 0.0f;
        for (int in_channel = 0; in_channel < weight_input_channels; ++in_channel) {
            for (int ky = 0; ky < weight_height; ++ky) {
                int image_y_base = y * stride + ky - padding;
                for (int kx = 0; kx < weight_width; ++kx) {
                    int image_x = x * stride + kx - padding;
                    if (image_y_base >= 0 && image_y_base < input_height && image_x >= 0 && image_x < input_width) {
                        int file_index = ((in_channel * input_height + image_y_base) * input_width + image_x);
                        int weight_index = ((((out_channel * weight_input_channels) + in_channel) * weight_height + ky) * weight_width + kx);
                        sum += input_data[file_index] * weight_data[weight_index];
                    }
                }
            }
        }
        // Add bias
        if (bias_data && out_channel < bias_number_of_elements) {
            sum += bias_data[out_channel];
        }
        // Apply ReLU if needed
        if (relu && sum < 0) {
            sum = 0.0f;
        }
        // Store result
        output_data[(out_channel * output_height + y) * output_width + x] = sum;
    }
}

void linearLayer(float* input_data, float* weights, float* bias, float* output_data, int input_size, int output_size) {
    for (int i = 0; i < output_size; ++i) {
        float temp = 0;
        for (int j = 0; j < input_size; ++j) {
            temp += input_data[j] * weights[i * input_size + j];
        }
        output_data[i] = temp + bias[i];
    }
}

int main(int argc, char** argv) {

    using std::chrono::duration_cast;
    using std::chrono::milliseconds;
    using std::chrono::microseconds;
    using std::chrono::steady_clock;

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_file_path>" << std::endl;
        return 1;
    }

    std::string filePath = argv[1];

    // Initialize parameters
    int image_input_channels = 3;
    int input_height = 32;
    int input_width = 32;
    int weight_output_channels = 64;
    int weight_input_channels = 3;
    int weight_height = 3;
    int weight_width = 3;
    int bias_number_of_elements = 64;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    int pool_size = 2; 
    int pool_stride = 2;
    bool relu = true;

    // Load the image data
    int imageDataSize = 3072;
    float* image_data = new float[imageDataSize];
    readDataFromFile(filePath, image_data, imageDataSize);

    // Load the weights and data for the first convolutional layer
    int weightDataSize = 1728;
    float* weight_data = new float[weightDataSize];
    readDataFromFile("data/dense/features_0_weight.txt", weight_data, weightDataSize);
    int biasDataSize = 64;
    float* bias_data = new float[biasDataSize];
    readDataFromFile("data/dense/features_0_bias.txt", bias_data, biasDataSize);

    // Load the weights and data for the second convolutional layer
    int secondWeightDataSize = 110592;
    float* second_weight_data = new float[secondWeightDataSize];
    readDataFromFile("data/dense/features_3_weight.txt", second_weight_data, secondWeightDataSize);
    int secondBiasDataSize = 192;
    float* second_bias_data = new float[secondBiasDataSize];
    readDataFromFile("data/dense/features_3_bias.txt", second_bias_data, secondBiasDataSize);

    // Load weights and biases for the third convolutional layer
    int thirdWeightDataSize = 663552;
    float* third_weight_data = new float[thirdWeightDataSize];
    readDataFromFile("data/dense/features_6_weight.txt", third_weight_data, thirdWeightDataSize);
    int thirdBiasDataSize = 384;
    float* third_bias_data = new float[thirdBiasDataSize];
    readDataFromFile("data/dense/features_6_bias.txt", third_bias_data, thirdBiasDataSize);

    // Load weights and biases for the fourth convolutional layer
    int fourthWeightDataSize = 884736;
    float* fourth_weight_data = new float[fourthWeightDataSize];
    readDataFromFile("data/dense/features_8_weight.txt", fourth_weight_data, fourthWeightDataSize);
    int fourthBiasDataSize = 256;
    float* fourth_bias_data = new float[fourthBiasDataSize];
    readDataFromFile("data/dense/features_8_bias.txt", fourth_bias_data, fourthBiasDataSize);

    // Load weights and biases for the fifth convolutional layer
    int fifthWeightDataSize = 589824;
    float* fifth_weight_data = new float[fifthWeightDataSize];
    readDataFromFile("data/dense/features_10_weight.txt", fifth_weight_data, fifthWeightDataSize);
    int fifthBiasDataSize = 256;
    float* fifth_bias_data = new float[fifthBiasDataSize];
    readDataFromFile("data/dense/features_10_bias.txt", fifth_bias_data, fifthBiasDataSize);

    // Load the weights and bias for the linear layer
    int linearWeightSize = 40960;
    float* linear_weight_data = new float[linearWeightSize];
    readDataFromFile("data/dense/classifier_weight.txt", linear_weight_data, linearWeightSize);
    int linearBiasSize = 10;
    float* linear_bias_data = new float[linearBiasSize];
    readDataFromFile("data/dense/classifier_bias.txt", linear_bias_data, linearBiasSize);

    // Allocate memory for convolution output
    int conv_output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int conv_output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    float* conv_output_data = new float[weight_output_channels * conv_output_height * conv_output_width];

    // Call the convolution function
    auto start_conv1 = steady_clock::now();
    conv2d(image_data, image_input_channels, input_height, input_width,
           weight_data, weight_output_channels, weight_input_channels, weight_height, weight_width,
           bias_data, bias_number_of_elements, kernel_size, stride, padding, relu, conv_output_data);
    auto end_conv1 = steady_clock::now();

    // Allocate memory for max pooling output
    int pooled_output_height = (conv_output_height - pool_size) / pool_stride + 1;
    int pooled_output_width = (conv_output_width - pool_size) / pool_stride + 1;
    float* maxpool_output_data = new float[weight_output_channels * pooled_output_height * pooled_output_width];

    // Apply max pooling on the convolution output
    auto start_maxpool1 = steady_clock::now();
    maxpool2d(conv_output_data, weight_output_channels, conv_output_height, conv_output_width, pool_size, pool_stride, maxpool_output_data);
    auto end_maxpool1 = steady_clock::now();

    // Allocate memory for the output of the second convolutional layer
    int second_conv_output_height = pooled_output_height;
    int second_conv_output_width = pooled_output_width;
    float* second_conv_output_data = new float[192 * second_conv_output_height * second_conv_output_width];

    // Call the convolution function for the second layer
    auto start_conv2 = steady_clock::now();
    conv2d(maxpool_output_data, 64, pooled_output_height, pooled_output_width,
           second_weight_data, 192, 64, 3, 3, 
           second_bias_data, 192, 3, 1, 1,
           true, second_conv_output_data);
    auto end_conv2 = steady_clock::now();

    // Allocate memory for the output of the second max pooling layer
    int second_pooled_output_height = (second_conv_output_height - pool_size) / pool_stride + 1;
    int second_pooled_output_width = (second_conv_output_width - pool_size) / pool_stride + 1;
    float* second_maxpool_output_data = new float[192 * second_pooled_output_height * second_pooled_output_width];

    // Apply the second max pooling on the second convolution output
    auto start_maxpool2 = steady_clock::now();
    maxpool2d(second_conv_output_data, 192, second_conv_output_height, second_conv_output_width, pool_size, pool_stride, second_maxpool_output_data);
    auto end_maxpool2 = steady_clock::now();

    // Allocate memory for the output of the third convolutional layer
    int third_conv_output_height = second_pooled_output_height;
    int third_conv_output_width = second_pooled_output_width;
    float* third_conv_output_data = new float[384 * third_conv_output_height * third_conv_output_width];

    // Apply the third convolution
    auto start_conv3 = steady_clock::now();
    conv2d(second_maxpool_output_data, 192, second_pooled_output_height, second_pooled_output_width,
           third_weight_data, 384, 192, 3, 3,
           third_bias_data, 384, 3, 1, 1,
           true, third_conv_output_data);
    auto end_conv3 = steady_clock::now();

    // Allocate memory for the output of the fourth convolutional layer
    int fourth_conv_output_channels = 256;
    int fourth_conv_output_height = third_conv_output_height;
    int fourth_conv_output_width = third_conv_output_width;
    float* fourth_conv_output_data = new float[fourth_conv_output_channels * fourth_conv_output_height * fourth_conv_output_width];

    // Call the convolution function for the fourth layer
    auto start_conv4 = steady_clock::now();
    conv2d(third_conv_output_data, 384, third_conv_output_height, third_conv_output_width,
           fourth_weight_data, fourth_conv_output_channels, 384, 3, 3,
           fourth_bias_data, fourth_conv_output_channels, 3, 1, 1,
           true, fourth_conv_output_data);
    auto end_conv4 = steady_clock::now();

    // Allocate memory for the output of the fifth convolutional layer
    int fifth_conv_output_channels = 256;
    int fifth_conv_output_height = fourth_conv_output_height;
    int fifth_conv_output_width = fourth_conv_output_width;
    float* fifth_conv_output_data = new float[fifth_conv_output_channels * fifth_conv_output_height * fifth_conv_output_width];

    // Apply the fifth convolution
    auto start_conv5 = steady_clock::now();
    conv2d(fourth_conv_output_data, 256, fourth_conv_output_height, fourth_conv_output_width,
           fifth_weight_data, fifth_conv_output_channels, 256, 3, 3,
           fifth_bias_data, fifth_conv_output_channels, 3, 1, 1,
           true, fifth_conv_output_data);
    auto end_conv5 = steady_clock::now();

    // Define parameters for the max pooling layer after the fifth convolution
    int pool_size_after_fifth = 2;
    int pool_stride_after_fifth = 2;

    // Calculate the output dimensions for the max pooling layer
    int fifth_pooled_output_height = (fifth_conv_output_height - pool_size_after_fifth) / pool_stride_after_fifth + 1;
    int fifth_pooled_output_width = (fifth_conv_output_width - pool_size_after_fifth) / pool_stride_after_fifth + 1;

    // Allocate memory for the output of the max pooling layer
    float* fifth_maxpool_output_data = new float[fifth_conv_output_channels * fifth_pooled_output_height * fifth_pooled_output_width];

    // Apply max pooling on the fifth convolution output
    auto start_maxpool3 = steady_clock::now();
    maxpool2d(fifth_conv_output_data, fifth_conv_output_channels, fifth_conv_output_height, fifth_conv_output_width,
              pool_size_after_fifth, pool_stride_after_fifth, fifth_maxpool_output_data);
    auto end_maxpool3 = steady_clock::now();

    // After the third max pooling layer, flatten the output
    int totalElements = fifth_conv_output_channels * fifth_pooled_output_height * fifth_pooled_output_width;
    float* flattened_output = new float[totalElements];

    int index = 0;
    for (int c = 0; c < fifth_conv_output_channels; ++c) {
        for (int h = 0; h < fifth_pooled_output_height; ++h) {
            for (int w = 0; w < fifth_pooled_output_width; ++w) {
                flattened_output[index++] = fifth_maxpool_output_data[(c * fifth_pooled_output_height + h) * fifth_pooled_output_width + w];
            }
        }
    }

    // Define the output size for the linear layer
    int linear_output_size = 10;  // Since the bias size is 10
    float* linear_output_data = new float[linear_output_size];

    // Call the linear layer function
    auto start_linear = steady_clock::now();
    linearLayer(flattened_output, linear_weight_data, linear_bias_data, linear_output_data, totalElements, linear_output_size);
    auto end_linear = steady_clock::now();

    // Find the index of the maximum element in the linear layer output
    int max_index = 0;
    float max_value = linear_output_data[0];
    for (int i = 1; i < linear_output_size; ++i) {
        if (linear_output_data[i] > max_value) {
            max_value = linear_output_data[i];
            max_index = i;
        }
    }

    // Map the index to the corresponding class and print the prediction
    std::cout << "Predicted Image: ";
    switch (max_index) {
        case 0: std::cout << "airplanes"; break;
        case 1: std::cout << "cars"; break;
        case 2: std::cout << "birds"; break;
        case 3: std::cout << "cats"; break;
        case 4: std::cout << "deer"; break;
        case 5: std::cout << "dogs"; break;
        case 6: std::cout << "frogs"; break;
        case 7: std::cout << "horses"; break;
        case 8: std::cout << "ships"; break;
        case 9: std::cout << "trucks"; break;
        default: std::cout << "Unknown"; break;
    }
    std::cout << std::endl;

    double conv1_time = duration_cast<milliseconds>(end_conv1 - start_conv1).count();
    double pool1_time = duration_cast<microseconds>(end_maxpool1 - start_maxpool1).count() / 1000.0;
    double conv2_time = duration_cast<milliseconds>(end_conv2 - start_conv2).count();
    double pool2_time = duration_cast<microseconds>(end_maxpool2 - start_maxpool2).count() / 1000.0;
    double conv3_time = duration_cast<milliseconds>(end_conv3 - start_conv3).count();
    double conv4_time = duration_cast<milliseconds>(end_conv4 - start_conv4).count();
    double conv5_time = duration_cast<milliseconds>(end_conv5 - start_conv5).count();
    double pool3_time = duration_cast<microseconds>(end_maxpool3 - start_maxpool3).count() / 1000.0;
    double linear_time = duration_cast<microseconds>(end_linear - start_linear).count() / 1000.0;

    double total_time = conv1_time + pool1_time + conv2_time + pool2_time +
                       conv3_time + conv4_time + conv5_time + pool3_time + linear_time;

    // Print layer-wise timing
    std::cout << "\nLayer-wise timing breakdown:" << std::endl;
    std::cout << "Conv1   : " << conv1_time << " ms" << std::endl;
    std::cout << "Pool1   : " << pool1_time << " ms" << std::endl;
    std::cout << "Conv2   : " << conv2_time << " ms" << std::endl;
    std::cout << "Pool2   : " << pool2_time << " ms" << std::endl;
    std::cout << "Conv3   : " << conv3_time << " ms" << std::endl;
    std::cout << "Conv4   : " << conv4_time << " ms" << std::endl;
    std::cout << "Conv5   : " << conv5_time << " ms" << std::endl;
    std::cout << "Pool3   : " << pool3_time << " ms" << std::endl;
    std::cout << "Linear  : " << linear_time << " ms" << std::endl;
    std::cout << "------------------------" << std::endl;
    std::cout << "Total   : " << total_time << " ms" << std::endl;

    // Free allocated memory
    delete[] image_data;
    delete[] weight_data;
    delete[] bias_data;
    delete[] conv_output_data;
    delete[] maxpool_output_data;
    delete[] second_weight_data;
    delete[] second_bias_data;
    delete[] second_conv_output_data;
    delete[] second_maxpool_output_data;
    delete[] third_weight_data;
    delete[] third_bias_data;
    delete[] fourth_weight_data;
    delete[] fourth_bias_data;
    delete[] fourth_conv_output_data;
    delete[] fifth_weight_data;
    delete[] fifth_bias_data;
    delete[] fifth_conv_output_data;
    delete[] flattened_output;

    return 0;
}
