#include<iostream>
#include<torch/torch.h>
#include<chrono>

torch::Tensor fspecial_gauss(int size, float sigma){
    torch::Tensor coords = torch::arange(size).to(torch::kFloat32);
    coords -= size/2;
    torch::Tensor g = torch::exp(-(coords * coords) / (2 * sigma * sigma));
    g /= g.sum();
    return g.unsqueeze(0).unsqueeze(0);
}

torch::Tensor gaussian_filter_2d(torch::Tensor inp, torch::Tensor win){
    auto in_shape = inp.sizes();
    auto win_shape = win.sizes();
    
    int c = in_shape[1];
    torch::Tensor out = inp.detach().clone();
    
    using namespace torch::nn::functional;
    out = conv2d(out, win.transpose(2, -1), Conv2dFuncOptions().stride(1).padding(0).groups(c));
    out = conv2d(out, win.transpose(3, -1), Conv2dFuncOptions().stride(1).padding(0).groups(c));
    return out;
}

std::pair<torch::Tensor, torch::Tensor> _ssim(
    torch::Tensor X,
    torch::Tensor Y,
    float data_range,
    torch::Tensor win,
    bool size_average=true,
    std::pair<float, float> K={0.01, 0.03}
){
    float K1 = K.first; float K2 = K.second;
    float C1 = (K1*data_range)*(K1*data_range);
    float C2 = (K2*data_range)*(K2*data_range);

    win = win.to(X.device()).to(torch::kFloat32);

    torch::Tensor mu1 = gaussian_filter_2d(X, win);
    torch::Tensor mu2 = gaussian_filter_2d(Y, win);

    auto mu1_sq = mu1.pow(2);
    auto mu2_sq = mu2.pow(2);
    auto mu1_mu2 = mu1*mu2;

    auto sigma1_sq = (gaussian_filter_2d(X * X, win) - mu1_sq);
    auto sigma2_sq = (gaussian_filter_2d(Y * Y, win) - mu2_sq);
    auto sigma12 =  (gaussian_filter_2d(X * Y, win) - mu1_mu2);

    auto cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2);
    auto ssim_map_temp = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1);
    auto ssim_map = torch::sqrt(torch::add(2-((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) - cs_map,0.001));
    auto ssim_per_channel = torch::flatten(ssim_map, 2).mean(-1);
    auto cs = torch::flatten(cs_map, 2).mean(-1);
    return {ssim_per_channel, cs};
}

torch::Tensor ssim(
    torch::Tensor X,
    torch::Tensor Y,
    float data_range=255.0,
    bool size_average=true,
    int win_size = 11,
    float win_sigma = 1.5,
    std::pair<float, float> K = {0.01, 0.03},
    bool nonnegative_ssim = false
){
    assert(X.sizes() == Y.sizes());

    for(int d = X.sizes().size()-1; d > 1; d--){
        X = X.squeeze(d);
        Y = Y.squeeze(d);
    }

    auto win = fspecial_gauss(win_size, win_sigma);
    win = win.repeat({X.sizes()[1], 1, 1, 1});

    auto res = _ssim(X, Y, data_range, win, false, K);
    torch::Tensor ssim_per_channel = res.first;
    torch::Tensor cs = res.second;
    if(nonnegative_ssim) ssim_per_channel = torch::relu(ssim_per_channel);

    if(size_average) return ssim_per_channel.mean();
    return ssim_per_channel.mean(1);
}

void test_gaussian(){
    torch::manual_seed(0);
    std::string device = "cpu";
    if(torch::cuda::is_available()) device = "cuda";
    torch::Tensor X = torch::rand({1, 3, 32, 32}).to(device);
    torch::Tensor win = fspecial_gauss(11, 1.5).to(device);
    win = win.repeat({3, 1, 1, 1});
    std::cout << gaussian_filter_2d(X, win);
}

void ssim_test(){
    torch::manual_seed(0);
    std::string device = "cpu";
    if(torch::cuda::is_available()) device = "cuda";
    torch::Tensor X = torch::rand({1, 3, 32, 32}).to(device);
    torch::Tensor Y = torch::rand({1, 3, 32, 32}).to(device);
    std::cout << ssim(X, Y);
}

int main(){
    // std::cout << fspecial_gauss(3, 0.5);
    // ssim_test();
    torch::manual_seed(0);
    std::string device = "cpu";
    if(torch::cuda::is_available()) device = "cuda";
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1; i++){
        torch::Tensor X = torch::rand({32, 3, 128, 128}).to(device);
        torch::Tensor Y = torch::rand({32, 3, 128, 128}).to(device);
        auto z = ssim(X, Y);
        std::cout << z;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << dur.count();
}