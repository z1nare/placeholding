// How to run:  nvcc -std=c++17 -o cuffting generate_spectrograms.cu -lcufft ; ./cuffting

#include <iostream>
#include <vector>
#include <cufft.h>
#include <cmath>
#include <string>
#include <filesystem>
#include <cuda_runtime.h>
#include <fstream>
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
    uint32_t sampleRate;         // <--- important
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char data[4];                // "data"
    uint32_t dataSize;
};


std::vector<float> load_wav_simple(const std::string &path, int &sr) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open " + path);

    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WavHeader));
    sr = header.sampleRate;

    int numSamples = header.dataSize / (header.bitsPerSample / 8);
    std::vector<int16_t> raw(numSamples);
    file.read(reinterpret_cast<char*>(raw.data()), header.dataSize);

    std::vector<float> waveform(numSamples);
    for (int i = 0; i < numSamples; i++)
        waveform[i] = raw[i] / 32768.0f;

    return waveform;
}

__global__ void magnitudeSpectrum(cufftComplex* fft_out, float* mag_out, int frame_size, int num_frames){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int num_bins = frame_size/2 +1;
    if (idx<num_frames*num_bins){
        int frame_idx = idx/num_bins;
        int bin_idx = idx%num_bins;
        cufftComplex c = fft_out[frame_idx*num_bins+bin_idx];
        mag_out[frame_idx*num_bins+bin_idx] = log1pf(sqrtf(c.x*c.x + c.y*c.y));
    }
}

void saveCSV(const std::string &filename, const std::vector<float>& data, int num_frames, int num_bins) {
    std::ofstream out(filename);
    for(int i=0;i<num_frames;i++){
        for(int j=0;j<num_bins;j++){
            out << data[i*num_bins+j];
            if(j<num_bins-1) out << ",";
        }
        out << "\n";
    }
}

int main() {
    std::string path = "/home/coder/project/data/bird";
    int frame_size = 512;
    int hop_size = 256;

    // Create output folder
    std::string out_folder = "processed_csvs";
    if (!std::filesystem::exists(out_folder)) {
        std::filesystem::create_directory(out_folder);
    }

    for (auto &p : std::filesystem::directory_iterator(path)) {
        if (p.path().extension() != ".wav") continue;

        int sr = 0;
        std::vector<float> waveform = load_wav_simple(p.path().string(), sr);

        int num_frames = (waveform.size() - frame_size) / hop_size + 1;
        std::vector<float> all_frames(num_frames * frame_size);

        // Slice waveform into frames
        for (int i = 0; i < num_frames; i++)
            for (int j = 0; j < frame_size; j++)
                all_frames[i*frame_size+j] = waveform[i*hop_size + j];

        // Pinned memory
        float* h_pinned;
        size_t n = all_frames.size()*sizeof(float);
        CUDA_CHECK(cudaMallocHost((void**)&h_pinned, n));
        memcpy(h_pinned, all_frames.data(), n);

        // Device memory
        float* d_frames;
        cufftComplex* d_fft_out;
        CUDA_CHECK(cudaMalloc((void**)&d_frames, n));
        CUDA_CHECK(cudaMalloc((void**)&d_fft_out, num_frames*(frame_size/2+1)*sizeof(cufftComplex)));
        CUDA_CHECK(cudaMemcpy(d_frames, h_pinned, n, cudaMemcpyHostToDevice));

        // cuFFT
        cufftHandle plan;
        cufftPlan1d(&plan, frame_size, CUFFT_R2C, num_frames);
        cufftExecR2C(plan, d_frames, d_fft_out);
        cudaDeviceSynchronize();

        // Magnitude
        float* d_mag;
        int num_bins = frame_size/2 + 1;
        CUDA_CHECK(cudaMalloc((void**)&d_mag, num_frames*num_bins*sizeof(float)));

        int threads = 256;
        int blocks = (num_frames*num_bins + threads - 1)/threads;
        magnitudeSpectrum<<<blocks, threads>>>(d_fft_out, d_mag, frame_size, num_frames);
        cudaDeviceSynchronize();

        std::vector<float> results(num_frames*num_bins);
        CUDA_CHECK(cudaMemcpy(results.data(), d_mag, num_frames*num_bins*sizeof(float), cudaMemcpyDeviceToHost));

        // Save CSV in processed_csvs folder
        std::string out_csv = out_folder + "/" + p.path().stem().string() + "_spectrogram.csv";
        saveCSV(out_csv, results, num_frames, num_bins);
        std::cout << "Saved " << out_csv << "\n";

        // Cleanup per file
        cudaFree(d_frames);
        cudaFree(d_fft_out);
        cudaFree(d_mag);
        cudaFreeHost(h_pinned);
        cufftDestroy(plan);
    }

    return 0;
}
