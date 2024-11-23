#include <iostream>
#include <fstream>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <vector>
#include <cmath>
#include <string>
#include <unordered_map>

// Optimized CSR format structure
struct CSRMatrix {
    std::vector<float> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
    std::unordered_map<int, int> value_index_map;
    int rows;
    int cols;
};

// Function to read data from file
void readDataFromFile(const std::string& filename, float* data, int& dataSize) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file - '" << filename << "'" << std::endl;
        return;
    }

    float value;
    int count = 0;
    while (file >> value) {
        ++count;
    }

    if (count != dataSize) {
        std::cerr << "Data size mismatch. Expected " << dataSize << " elements, but file contains " << count << "." << std::endl;
        return;
    }

    file.clear();
    file.seekg(0, std::ios::beg);

    int index = 0;
    while (file >> value) {
        data[index++] = value;
    }

    file.close();
}

// Optimized dense to CSR conversion
CSRMatrix denseToCsr(float* dense_data, int rows, int cols, float threshold = 1e-6) {
    CSRMatrix csr;
    csr.rows = rows;
    csr.cols = cols;
    csr.row_ptr.push_back(0);
    
    int value_idx = 0;
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            float val = dense_data[i * cols + j];
            if(std::abs(val) > threshold) {
                csr.values.push_back(val);
                csr.col_indices.push_back(j);
                csr.value_index_map[i * cols + j] = value_idx++;
            }
        }
        csr.row_ptr.push_back(csr.values.size());
    }
    return csr;
}

// Optimized sparse convolution
void sparseConv2d(const CSRMatrix& sparse_input, int input_height, int input_width,
                  const CSRMatrix& sparse_weights, int weight_output_channels,
                  float* bias_data, int kernel_size, int stride, int padding,
                  bool relu, float* output_data) {
    
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    int output_size = weight_output_channels * output_height * output_width;
    
    std::fill(output_data, output_data + output_size, 0.0f);

    struct PaddedPosition {
        int x, y, channel;
        float value;
    };
    
    std::vector<std::vector<PaddedPosition>> padded_positions(
        (input_height + 2 * padding) * (input_width + 2 * padding)
    );

    // Populate padded positions
    for(int in_c = 0; in_c < sparse_input.rows; ++in_c) {
        for(int idx = sparse_input.row_ptr[in_c]; idx < sparse_input.row_ptr[in_c + 1]; ++idx) {
            int orig_pos = sparse_input.col_indices[idx];
            int orig_y = orig_pos / input_width;
            int orig_x = orig_pos % input_width;
            
            int padded_y = orig_y + padding;
            int padded_x = orig_x + padding;
            
            padded_positions[padded_y * (input_width + 2 * padding) + padded_x].push_back(
                {padded_x, padded_y, in_c, sparse_input.values[idx]}
            );
        }
    }

    // Pre-compute weight indices
    std::vector<std::vector<std::pair<int, float>>> weight_values(weight_output_channels);
    for(int out_c = 0; out_c < weight_output_channels; ++out_c) {
        for(int w_idx = sparse_weights.row_ptr[out_c]; 
            w_idx < sparse_weights.row_ptr[out_c + 1]; ++w_idx) {
            weight_values[out_c].push_back({
                sparse_weights.col_indices[w_idx],
                sparse_weights.values[w_idx]
            });
        }
    }

    // Main convolution loop
    for(int out_c = 0; out_c < weight_output_channels; ++out_c) {
        float bias = bias_data ? bias_data[out_c] : 0.0f;
        const auto& channel_weights = weight_values[out_c];
        
        for(int h = 0; h < output_height; ++h) {
            for(int w = 0; w < output_width; ++w) {
                float sum = bias;
                int output_idx = (out_c * output_height + h) * output_width + w;
                
                for(const auto& weight_pair : channel_weights) {
                    int weight_offset = weight_pair.first;
                    int in_c = weight_offset / (kernel_size * kernel_size);
                    int ky = (weight_offset % (kernel_size * kernel_size)) / kernel_size;
                    int kx = weight_offset % kernel_size;
                    
                    int in_y = h * stride + ky;
                    int in_x = w * stride + kx;
                    
                    const auto& positions = padded_positions[in_y * (input_width + 2 * padding) + in_x];
                    for(const auto& pos : positions) {
                        if(pos.channel == in_c) {
                            sum += pos.value * weight_pair.second;
                            break;
                        }
                    }
                }
                
                output_data[output_idx] = relu && sum < 0 ? 0.0f : sum;
            }
        }
    }
}

// Optimized sparse max pooling
void sparseMaxPool2d(const CSRMatrix& sparse_input, int input_channels, int input_height,
                     int input_width, int pool_size, int stride, float* output_data) {
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;
    int output_size = input_channels * output_height * output_width;
    
    std::fill(output_data, output_data + output_size, -FLT_MAX);
    
    struct OutputPosition {
        int x, y;
        float value;
    };
    
    std::vector<std::vector<OutputPosition>> output_positions(input_channels);
    
    // Pre-compute output positions
    for(int c = 0; c < input_channels; ++c) {
        for(int row = sparse_input.row_ptr[c]; row < sparse_input.row_ptr[c + 1]; ++row) {
            int col = sparse_input.col_indices[row];
            int in_x = col % input_width;
            int in_y = (col / input_width) % input_height;
            float value = sparse_input.values[row];
            
            int out_start_x = std::max(0, (in_x - pool_size + stride) / stride);
            int out_start_y = std::max(0, (in_y - pool_size + stride) / stride);
            int out_end_x = std::min(output_width - 1, in_x / stride);
            int out_end_y = std::min(output_height - 1, in_y / stride);
            
            for(int out_y = out_start_y; out_y <= out_end_y; ++out_y) {
                for(int out_x = out_start_x; out_x <= out_end_x; ++out_x) {
                    if(in_x >= out_x * stride && in_x < out_x * stride + pool_size &&
                       in_y >= out_y * stride && in_y < out_y * stride + pool_size) {
                        output_positions[c].push_back({out_x, out_y, value});
                    }
                }
            }
        }
    }
    
    // Update output values
    for(int c = 0; c < input_channels; ++c) {
        for(const auto& pos : output_positions[c]) {
            int out_idx = (c * output_height + pos.y) * output_width + pos.x;
            output_data[out_idx] = std::max(output_data[out_idx], pos.value);
        }
    }
    
    // Replace -FLT_MAX with zeros
    for(int i = 0; i < output_size; ++i) {
        if(output_data[i] == -FLT_MAX) {
            output_data[i] = 0.0f;
        }
    }
}

// Optimized sparse linear layer
void sparseLinearLayer(const CSRMatrix& sparse_input, const CSRMatrix& sparse_weights,
                      float* bias, float* output_data, int output_size) {
    std::vector<std::pair<int, float>> input_values;
    for(int in_idx = sparse_input.row_ptr[0]; in_idx < sparse_input.row_ptr[1]; ++in_idx) {
        input_values.push_back({
            sparse_input.col_indices[in_idx],
            sparse_input.values[in_idx]
        });
    }
    
    // Initialize with bias
    for(int i = 0; i < output_size; ++i) {
        output_data[i] = bias[i];
    }
    
    // Process weights
    for(int i = 0; i < output_size; ++i) {
        const auto& row_start = sparse_weights.row_ptr[i];
        const auto& row_end = sparse_weights.row_ptr[i + 1];
        
        for(const auto& input_pair : input_values) {
            auto weight_it = std::lower_bound(
                sparse_weights.col_indices.begin() + row_start,
                sparse_weights.col_indices.begin() + row_end,
                input_pair.first
            );
            
            if(weight_it != sparse_weights.col_indices.begin() + row_end && 
               *weight_it == input_pair.first) {
                int weight_idx = weight_it - sparse_weights.col_indices.begin();
                output_data[i] += input_pair.second * sparse_weights.values[weight_idx];
            }
        }
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
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    int pool_size = 2;
    int pool_stride = 2;
    bool relu = true;
    float sparsity_threshold = 1e-6;

    // Load and process initial data
    int imageDataSize = image_input_channels * input_height * input_width;
    float* image_data = new float[imageDataSize];
    readDataFromFile(filePath, image_data, imageDataSize);
    CSRMatrix sparse_image = denseToCsr(image_data, image_input_channels,
                                      input_height * input_width, sparsity_threshold);

    // First convolution layer
    int weightDataSize = weight_output_channels * weight_input_channels * kernel_size * kernel_size;
    float* weight_data = new float[weightDataSize];
    readDataFromFile("data/sparse/features_0_weight.txt", weight_data, weightDataSize);
    CSRMatrix sparse_weights = denseToCsr(weight_data, weight_output_channels,
                                        weight_input_channels * kernel_size * kernel_size,
                                        sparsity_threshold);

    float* bias_data = new float[weight_output_channels];
    readDataFromFile("data/sparse/features_0_bias.txt", bias_data, weight_output_channels);

    int conv_output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int conv_output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    float* conv_output_data = new float[weight_output_channels * conv_output_height * conv_output_width];

    // Time and execute first convolution
    auto start_conv1 = steady_clock::now();
    sparseConv2d(sparse_image, input_height, input_width,
                 sparse_weights, weight_output_channels,
                 bias_data, kernel_size, stride, padding,
                 relu, conv_output_data);
    auto end_conv1 = steady_clock::now();

    CSRMatrix sparse_conv_output = denseToCsr(conv_output_data,
                                            weight_output_channels,
                                            conv_output_height * conv_output_width,
                                            sparsity_threshold);

    // First max pooling layer
    int pooled_height = (conv_output_height - pool_size) / pool_stride + 1;
    int pooled_width = (conv_output_width - pool_size) / pool_stride + 1;
    float* pool_output_data = new float[weight_output_channels * pooled_height * pooled_width];

    auto start_pool1 = steady_clock::now();
    sparseMaxPool2d(sparse_conv_output, weight_output_channels,
                    conv_output_height, conv_output_width,
                    pool_size, pool_stride, pool_output_data);
    auto end_pool1 = steady_clock::now();


    // Second convolutional layer
    int second_weight_channels = 192;
    int second_weightDataSize = second_weight_channels * weight_output_channels * kernel_size * kernel_size;
    float* second_weight_data = new float[second_weightDataSize];
    readDataFromFile("data/sparse/features_3_weight.txt", second_weight_data, second_weightDataSize);
    CSRMatrix sparse_second_weights = denseToCsr(second_weight_data, second_weight_channels,
                                               weight_output_channels * kernel_size * kernel_size,
                                               sparsity_threshold);

    float* second_bias_data = new float[second_weight_channels];
    readDataFromFile("data/sparse/features_3_bias.txt", second_bias_data, second_weight_channels);

    CSRMatrix sparse_pool_output = denseToCsr(pool_output_data, weight_output_channels,
                                            pooled_height * pooled_width, sparsity_threshold);

    float* second_conv_output = new float[second_weight_channels * pooled_height * pooled_width];

    auto start_conv2 = steady_clock::now();
    sparseConv2d(sparse_pool_output, pooled_height, pooled_width,
                 sparse_second_weights, second_weight_channels,
                 second_bias_data, kernel_size, stride, padding,
                 relu, second_conv_output);
    auto end_conv2 = steady_clock::now();

    // Second max pooling
    int second_pooled_height = (pooled_height - pool_size) / pool_stride + 1;
    int second_pooled_width = (pooled_width - pool_size) / pool_stride + 1;
    float* second_pool_output = new float[second_weight_channels * second_pooled_height * second_pooled_width];

    CSRMatrix sparse_second_conv = denseToCsr(second_conv_output, second_weight_channels,
                                            pooled_height * pooled_width, sparsity_threshold);

    auto start_pool2 = steady_clock::now();
    sparseMaxPool2d(sparse_second_conv, second_weight_channels,
                    pooled_height, pooled_width,
                    pool_size, pool_stride, second_pool_output);
    auto end_pool2 = steady_clock::now();

    // Third convolutional layer
    int third_weight_channels = 384;
    int third_weightDataSize = third_weight_channels * second_weight_channels * kernel_size * kernel_size;
    float* third_weight_data = new float[third_weightDataSize];
    readDataFromFile("data/sparse/features_6_weight.txt", third_weight_data, third_weightDataSize);
    CSRMatrix sparse_third_weights = denseToCsr(third_weight_data, third_weight_channels,
                                              second_weight_channels * kernel_size * kernel_size,
                                              sparsity_threshold);

    float* third_bias_data = new float[third_weight_channels];
    readDataFromFile("data/sparse/features_6_bias.txt", third_bias_data, third_weight_channels);

    CSRMatrix sparse_second_pool = denseToCsr(second_pool_output, second_weight_channels,
                                            second_pooled_height * second_pooled_width,
                                            sparsity_threshold);

    float* third_conv_output = new float[third_weight_channels * second_pooled_height * second_pooled_width];

    auto start_conv3 = steady_clock::now();
    sparseConv2d(sparse_second_pool, second_pooled_height, second_pooled_width,
                 sparse_third_weights, third_weight_channels,
                 third_bias_data, kernel_size, stride, padding,
                 relu, third_conv_output);
    auto end_conv3 = steady_clock::now();

    // Fourth convolutional layer
    int fourth_weight_channels = 256;
    int fourth_weightDataSize = fourth_weight_channels * third_weight_channels * kernel_size * kernel_size;
    float* fourth_weight_data = new float[fourth_weightDataSize];
    readDataFromFile("data/sparse/features_8_weight.txt", fourth_weight_data, fourth_weightDataSize);
    CSRMatrix sparse_fourth_weights = denseToCsr(fourth_weight_data, fourth_weight_channels,
                                               third_weight_channels * kernel_size * kernel_size,
                                               sparsity_threshold);

    float* fourth_bias_data = new float[fourth_weight_channels];
    readDataFromFile("data/sparse/features_8_bias.txt", fourth_bias_data, fourth_weight_channels);

    CSRMatrix sparse_third_conv = denseToCsr(third_conv_output, third_weight_channels,
                                           second_pooled_height * second_pooled_width,
                                           sparsity_threshold);

    float* fourth_conv_output = new float[fourth_weight_channels * second_pooled_height * second_pooled_width];

    auto start_conv4 = steady_clock::now();
    sparseConv2d(sparse_third_conv, second_pooled_height, second_pooled_width,
                 sparse_fourth_weights, fourth_weight_channels,
                 fourth_bias_data, kernel_size, stride, padding,
                 relu, fourth_conv_output);
    auto end_conv4 = steady_clock::now();

    // Fifth convolutional layer
    int fifth_weight_channels = 256;
    int fifth_weightDataSize = fifth_weight_channels * fourth_weight_channels * kernel_size * kernel_size;
    float* fifth_weight_data = new float[fifth_weightDataSize];
    readDataFromFile("data/sparse/features_10_weight.txt", fifth_weight_data, fifth_weightDataSize);
    CSRMatrix sparse_fifth_weights = denseToCsr(fifth_weight_data, fifth_weight_channels,
                                              fourth_weight_channels * kernel_size * kernel_size,
                                              sparsity_threshold);

    float* fifth_bias_data = new float[fifth_weight_channels];
    readDataFromFile("data/sparse/features_10_bias.txt", fifth_bias_data, fifth_weight_channels);

    CSRMatrix sparse_fourth_conv = denseToCsr(fourth_conv_output, fourth_weight_channels,
                                            second_pooled_height * second_pooled_width,
                                            sparsity_threshold);

    float* fifth_conv_output = new float[fifth_weight_channels * second_pooled_height * second_pooled_width];

    auto start_conv5 = steady_clock::now();
    sparseConv2d(sparse_fourth_conv, second_pooled_height, second_pooled_width,
                 sparse_fifth_weights, fifth_weight_channels,
                 fifth_bias_data, kernel_size, stride, padding,
                 relu, fifth_conv_output);
    auto end_conv5 = steady_clock::now();

    // Final max pooling
    int final_pooled_height = (second_pooled_height - pool_size) / pool_stride + 1;
    int final_pooled_width = (second_pooled_width - pool_size) / pool_stride + 1;
    float* final_pool_output = new float[fifth_weight_channels * final_pooled_height * final_pooled_width];

    CSRMatrix sparse_fifth_conv = denseToCsr(fifth_conv_output, fifth_weight_channels,
                                           second_pooled_height * second_pooled_width,
                                           sparsity_threshold);

    auto start_pool3 = steady_clock::now();
    sparseMaxPool2d(sparse_fifth_conv, fifth_weight_channels,
                    second_pooled_height, second_pooled_width,
                    pool_size, pool_stride, final_pool_output);
    auto end_pool3 = steady_clock::now();

    // Flatten the output
    int flattened_size = fifth_weight_channels * final_pooled_height * final_pooled_width;
    float* flattened_output = new float[flattened_size];

    int flatten_idx = 0;
    for (int c = 0; c < fifth_weight_channels; ++c) {
        for (int h = 0; h < final_pooled_height; ++h) {
            for (int w = 0; w < final_pooled_width; ++w) {
                flattened_output[flatten_idx++] = final_pool_output[
                    (c * final_pooled_height + h) * final_pooled_width + w];
            }
        }
    }

    // Linear layer
    int linear_output_size = 10;
    int linear_weight_size = flattened_size * linear_output_size;
    float* linear_weight_data = new float[linear_weight_size];
    readDataFromFile("data/sparse/classifier_weight.txt", linear_weight_data, linear_weight_size);
    CSRMatrix sparse_linear_weights = denseToCsr(linear_weight_data, linear_output_size,
                                               flattened_size, sparsity_threshold);

    float* linear_bias_data = new float[linear_output_size];
    readDataFromFile("data/sparse/classifier_bias.txt", linear_bias_data, linear_output_size);

    float* output_data = new float[linear_output_size];
    CSRMatrix sparse_flattened = denseToCsr(flattened_output, 1, flattened_size, sparsity_threshold);

    auto start_linear = steady_clock::now();
    sparseLinearLayer(sparse_flattened, sparse_linear_weights,
                      linear_bias_data, output_data,
                      linear_output_size);
    auto end_linear = steady_clock::now();

    // Find prediction
    int max_index = 0;
    float max_value = output_data[0];
    for (int i = 1; i < linear_output_size; ++i) {
        if (output_data[i] > max_value) {
            max_value = output_data[i];
            max_index = i;
        }
    }

    // Print prediction
    std::cout << "Predicted Class: ";
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

    // Calculate and print timing
    double conv1_time = duration_cast<milliseconds>(end_conv1 - start_conv1).count();
    double pool1_time = duration_cast<microseconds>(end_pool1 - start_pool1).count() / 1000.0;
    double conv2_time = duration_cast<milliseconds>(end_conv2 - start_conv2).count();
    double pool2_time = duration_cast<microseconds>(end_pool2 - start_pool2).count() / 1000.0;
    double conv3_time = duration_cast<milliseconds>(end_conv3 - start_conv3).count();
    double conv4_time = duration_cast<milliseconds>(end_conv4 - start_conv4).count();
    double conv5_time = duration_cast<milliseconds>(end_conv5 - start_conv5).count();
    double pool3_time = duration_cast<microseconds>(end_pool3 - start_pool3).count() / 1000.0;
    double linear_time = duration_cast<microseconds>(end_linear - start_linear).count() / 1000.0;

    double total_time = conv1_time + pool1_time + conv2_time + pool2_time +
                       conv3_time + conv4_time + conv5_time + pool3_time + linear_time;

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

    // Cleanup
    delete[] image_data;
    delete[] weight_data;
    delete[] bias_data;
    delete[] conv_output_data;
    delete[] pool_output_data;
    delete[] second_weight_data;
    delete[] second_bias_data;
    delete[] second_conv_output;
    delete[] second_pool_output;
    delete[] third_weight_data;
    delete[] third_bias_data;
    delete[] third_conv_output;
    delete[] fourth_weight_data;
    delete[] fourth_bias_data;
    delete[] fourth_conv_output;
    delete[] fifth_weight_data;
    delete[] fifth_bias_data;
    delete[] fifth_conv_output;
    delete[] final_pool_output;
    delete[] flattened_output;
    delete[] linear_weight_data;
    delete[] linear_bias_data;
    delete[] output_data;

    return 0;
}
