#include <iostream>
#include <fstream>
#include <cfloat>
#include <chrono>

// Maximum sizes for static arrays
const int MAX_NNZ_CONV1 = 1728;    // 3*3*3*64
const int MAX_NNZ_CONV2 = 110592;  // 3*3*64*192
const int MAX_NNZ_CONV3 = 663552;  // 3*3*192*384
const int MAX_NNZ_CONV4 = 884736;  // 3*3*384*256
const int MAX_NNZ_CONV5 = 589824;  // 3*3*256*256
const int MAX_NNZ_LINEAR = 40960;  // 256*4*4*10

// Static arrays for CSR format
struct CSRMatrix {
    const float* values;
    const int* row_ptr;
    const int* col_idx;
    int rows;
    int cols;
    int nnz;
};

// Static arrays for all layers
float conv1_values[MAX_NNZ_CONV1];
int conv1_row_ptr[65];  // 64 + 1
int conv1_col_idx[MAX_NNZ_CONV1];

float conv2_values[MAX_NNZ_CONV2];
int conv2_row_ptr[193];  // 192 + 1
int conv2_col_idx[MAX_NNZ_CONV2];

float conv3_values[MAX_NNZ_CONV3];
int conv3_row_ptr[385];  // 384 + 1
int conv3_col_idx[MAX_NNZ_CONV3];

float conv4_values[MAX_NNZ_CONV4];
int conv4_row_ptr[257];  // 256 + 1
int conv4_col_idx[MAX_NNZ_CONV4];

float conv5_values[MAX_NNZ_CONV5];
int conv5_row_ptr[257];  // 256 + 1
int conv5_col_idx[MAX_NNZ_CONV5];

float linear_values[MAX_NNZ_LINEAR];
int linear_row_ptr[11];  // 10 + 1
int linear_col_idx[MAX_NNZ_LINEAR];

// Static arrays for intermediate results
float conv1_output[64 * 32 * 32];
float pool1_output[64 * 16 * 16];
float conv2_output[192 * 16 * 16];
float pool2_output[192 * 8 * 8];
float conv3_output[384 * 8 * 8];
float conv4_output[256 * 8 * 8];
float conv5_output[256 * 8 * 8];
float pool3_output[256 * 4 * 4];
float linear_output[10];

// Static arrays for biases
float conv1_bias[64];
float conv2_bias[192];
float conv3_bias[384];
float conv4_bias[256];
float conv5_bias[256];
float linear_bias[10];

void readDataFromFile(const char* filename, float* data, int maxSize) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file - '" << filename << "'" << std::endl;
        return;
    }

    // Zero initialize the entire array
    for (int i = 0; i < maxSize; ++i) {
        data[i] = 0.0f;
    }

    // Read available values
    float value;
    int count = 0;
    while (file >> value && count < maxSize) {
        data[count++] = value;
    }

    file.close();
}

void readIntDataFromFile(const char* filename, int* data, int maxSize) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file - '" << filename << "'" << std::endl;
        return;
    }

    int value;
    int count = 0;
    while (file >> value && count < maxSize) {
        data[count++] = value;
    }

    file.close();
}

void readCSRFromFiles(const char* values_file, const char* row_ptr_file, const char* col_idx_file,
                     float* values, int* row_ptr, int* col_idx,
                     int nnz, int rows) {
    readDataFromFile(values_file, values, nnz);
    
    std::ifstream row_file(row_ptr_file);
    for (int i = 0; i <= rows; ++i) {
        row_file >> row_ptr[i];
    }
    row_file.close();

    std::ifstream col_file(col_idx_file);
    for (int i = 0; i < nnz; ++i) {
        col_file >> col_idx[i];
    }
    col_file.close();
}

void maxpool2d(const float* input_data, int input_channels, int input_height, int input_width,
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

void conv2d_sparse(const float* input_data, int image_input_channels, int input_height, int input_width,
                  const CSRMatrix& weight_matrix, const float* bias_data, int bias_size,
                  int kernel_size, int stride, int padding, bool relu, float* output_data) {
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    int output_channels = weight_matrix.rows;
    int spatial_size = kernel_size * kernel_size * image_input_channels;

    // Zero initialize output
    int output_size = output_channels * output_height * output_width;
    for (int i = 0; i < output_size; ++i) {
        output_data[i] = 0.0f;
    }

    for (int out_c = 0; out_c < output_channels; ++out_c) {
        int row_start = weight_matrix.row_ptr[out_c];
        int row_end = weight_matrix.row_ptr[out_c + 1];

        for (int oh = 0; oh < output_height; ++oh) {
            for (int ow = 0; ow < output_width; ++ow) {
                float sum = 0.0f;

                for (int nz_idx = row_start; nz_idx < row_end; ++nz_idx) {
                    int flat_kernel_idx = weight_matrix.col_idx[nz_idx];
                    float weight_value = weight_matrix.values[nz_idx];

                    int in_c = flat_kernel_idx / (kernel_size * kernel_size);
                    int rem = flat_kernel_idx % (kernel_size * kernel_size);
                    int ky = rem / kernel_size;
                    int kx = rem % kernel_size;

                    int ih = oh * stride + ky - padding;
                    int iw = ow * stride + kx - padding;

                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                        int input_idx = (in_c * input_height + ih) * input_width + iw;
                        sum += input_data[input_idx] * weight_value;
                    }
                }

                if (bias_data && out_c < bias_size) {
                    sum += bias_data[out_c];
                }

                if (relu && sum < 0) {
                    sum = 0.0f;
                }

                output_data[(out_c * output_height + oh) * output_width + ow] = sum;
            }
        }
    }
}

void linearLayer_sparse(const float* input_data, const CSRMatrix& weight_matrix,
                       const float* bias_data, float* output_data) {
    for (int i = 0; i < weight_matrix.rows; ++i) {
        float sum = 0.0f;
        
        for (int nz_idx = weight_matrix.row_ptr[i]; nz_idx < weight_matrix.row_ptr[i + 1]; ++nz_idx) {
            int col = weight_matrix.col_idx[nz_idx];
            sum += input_data[col] * weight_matrix.values[nz_idx];
        }
        
        output_data[i] = sum + bias_data[i];
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

    // Load image data
    float image_data[3 * 32 * 32];
    readDataFromFile(argv[1], image_data, 3072);

    // Load CSR data for all layers
    readCSRFromFiles("data/sparse/conv1_values.txt", "data/sparse/conv1_row_ptr.txt", 
                    "data/sparse/conv1_col_idx.txt", conv1_values, conv1_row_ptr, conv1_col_idx, 
                    MAX_NNZ_CONV1, 64);

    readCSRFromFiles("data/sparse/conv2_values.txt", "data/sparse/conv2_row_ptr.txt",
                    "data/sparse/conv2_col_idx.txt", conv2_values, conv2_row_ptr, conv2_col_idx,
                    MAX_NNZ_CONV2, 192);

    readCSRFromFiles("data/sparse/conv3_values.txt", "data/sparse/conv3_row_ptr.txt",
                    "data/sparse/conv3_col_idx.txt", conv3_values, conv3_row_ptr, conv3_col_idx,
                    MAX_NNZ_CONV3, 384);

    readCSRFromFiles("data/sparse/conv4_values.txt", "data/sparse/conv4_row_ptr.txt",
                    "data/sparse/conv4_col_idx.txt", conv4_values, conv4_row_ptr, conv4_col_idx,
                    MAX_NNZ_CONV4, 256);

    readCSRFromFiles("data/sparse/conv5_values.txt", "data/sparse/conv5_row_ptr.txt",
                    "data/sparse/conv5_col_idx.txt", conv5_values, conv5_row_ptr, conv5_col_idx,
                    MAX_NNZ_CONV5, 256);

    readCSRFromFiles("data/sparse/linear_values.txt", "data/sparse/linear_row_ptr.txt",
                    "data/sparse/linear_col_idx.txt", linear_values, linear_row_ptr, linear_col_idx,
                    MAX_NNZ_LINEAR, 10);

    // Load biases
    readDataFromFile("data/sparse/conv1_bias.txt", conv1_bias, 64);
    readDataFromFile("data/sparse/conv2_bias.txt", conv2_bias, 192);
    readDataFromFile("data/sparse/conv3_bias.txt", conv3_bias, 384);
    readDataFromFile("data/sparse/conv4_bias.txt", conv4_bias, 256);
    readDataFromFile("data/sparse/conv5_bias.txt", conv5_bias, 256);
    readDataFromFile("data/sparse/linear_bias.txt", linear_bias, 10);

    // Create CSR matrices
    CSRMatrix conv1_weights = {conv1_values, conv1_row_ptr, conv1_col_idx, 64, 27, MAX_NNZ_CONV1};
    CSRMatrix conv2_weights = {conv2_values, conv2_row_ptr, conv2_col_idx, 192, 576, MAX_NNZ_CONV2};
    CSRMatrix conv3_weights = {conv3_values, conv3_row_ptr, conv3_col_idx, 384, 1728, MAX_NNZ_CONV3};
    CSRMatrix conv4_weights = {conv4_values, conv4_row_ptr, conv4_col_idx, 256, 3456, MAX_NNZ_CONV4};
    CSRMatrix conv5_weights = {conv5_values, conv5_row_ptr, conv5_col_idx, 256, 2304, MAX_NNZ_CONV5};
    CSRMatrix linear_weights = {linear_values, linear_row_ptr, linear_col_idx, 10, 4096, MAX_NNZ_LINEAR};

    // First convolution layer
    auto start_conv1 = steady_clock::now();
    conv2d_sparse(image_data, 3, 32, 32, conv1_weights, conv1_bias, 64, 3, 1, 1, true, conv1_output);
    auto end_conv1 = steady_clock::now();

    // First max pooling layer
    auto start_pool1 = steady_clock::now();
    maxpool2d(conv1_output, 64, 32, 32, 2, 2, pool1_output);
    auto end_pool1 = steady_clock::now();

    // Second convolution layer
    auto start_conv2 = steady_clock::now();
    conv2d_sparse(pool1_output, 64, 16, 16, conv2_weights, conv2_bias, 192, 3, 1, 1, true, conv2_output);
    auto end_conv2 = steady_clock::now();

    // Second max pooling layer
    auto start_pool2 = steady_clock::now();
    maxpool2d(conv2_output, 192, 16, 16, 2, 2, pool2_output);
    auto end_pool2 = steady_clock::now();

    // Third convolution layer
    auto start_conv3 = steady_clock::now();
    conv2d_sparse(pool2_output, 192, 8, 8, conv3_weights, conv3_bias, 384, 3, 1, 1, true, conv3_output);
    auto end_conv3 = steady_clock::now();

    // Fourth convolution layer
    auto start_conv4 = steady_clock::now();
    conv2d_sparse(conv3_output, 384, 8, 8, conv4_weights, conv4_bias, 256, 3, 1, 1, true, conv4_output);
    auto end_conv4 = steady_clock::now();

    // Fifth convolution layer
    auto start_conv5 = steady_clock::now();
    conv2d_sparse(conv4_output, 256, 8, 8, conv5_weights, conv5_bias, 256, 3, 1, 1, true, conv5_output);
    auto end_conv5 = steady_clock::now();

    // Third max pooling layer
    auto start_pool3 = steady_clock::now();
    maxpool2d(conv5_output, 256, 8, 8, 2, 2, pool3_output);
    auto end_pool3 = steady_clock::now();

    // Linear layer
    auto start_linear = steady_clock::now();
    linearLayer_sparse(pool3_output, linear_weights, linear_bias, linear_output);
    auto end_linear = steady_clock::now();

    // Find the index of the maximum element in the linear layer output
    int max_index = 0;
    float max_value = linear_output[0];
    for (int i = 1; i < 10; ++i) {
        if (linear_output[i] > max_value) {
            max_value = linear_output[i];
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

    // Calculate timing
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

    return 0;
}
