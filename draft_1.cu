#include <iostream>
#include <vector>
#include <cufft.h>
#include <math.h>
#include <string>
#include <filesystem>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <cstdint>



#define CUDA_CHECK(call) \
    { cudaError_t err = (call); \
      if(err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); } }


struct WavHeader {
    char riff[4];                // "RIFF"
    uint32_t chunkSize;
    char wave[4];                // "WAVE"
    char fmt[4];                 // "fmt "
    uint32_t subchunk1Size;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char data[4];                // "data"
    uint32_t dataSize;
};

std::vector<float> load_wav_simple(const std::string &path, int &sr) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WavHeader));

    sr = header.sampleRate;

    int numSamples = header.dataSize / (header.bitsPerSample / 8);
    std::vector<int16_t> raw(numSamples);
    file.read(reinterpret_cast<char*>(raw.data()), header.dataSize);

    // Convert to normalized float in [-1,1]
    std::vector<float> waveform(numSamples);
    for (int i = 0; i < numSamples; i++) {
        waveform[i] = raw[i] / 32768.0f;
    }

    return waveform;
}



int main()
{
    std::string path = "/data/bird";
    int sr = 0;
    std::string wav_path = path + "/1002.wav";
    std::vector<float> waveform = load_wav_simple(wav_path, sr);

    std::cout << "Loaded " << wav_path 
              << " with " << waveform.size() 
              << " samples at " << sr << " Hz" << std::endl;
    std::cout << "First 10 samples (CPU):" << std::endl;
    for (int i = 0; i < 10 && i < waveform.size(); i++) {
        std::cout << waveform[i] << " ";
    }
std::cout << std::endl;
    // Step 2: Allocate pinned host memory
    float *h_pinned;
    size_t nbytes = waveform.size() * sizeof(float);
    CUDA_CHECK(cudaMallocHost((void**)&h_pinned, nbytes));

    // Copy from std::vector to pinned buffer
    memcpy(h_pinned, waveform.data(), nbytes);

    // Allocate device memory
    float *d_waveform;
    CUDA_CHECK(cudaMalloc((void**)&d_waveform, nbytes));

    // Async copy from pinned host → device
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(d_waveform, h_pinned, nbytes, cudaMemcpyHostToDevice, stream));

    // Wait for transfer to finish
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> check_back(10);
    CUDA_CHECK(cudaMemcpy(check_back.data(), d_waveform, 10 * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "First 10 samples (back from GPU):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << check_back[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Copied waveform to GPU memory" << std::endl;

    // Cleanup
    cudaFree(d_waveform);
    cudaFreeHost(h_pinned);
    cudaStreamDestroy(stream);

    return 0;
}
