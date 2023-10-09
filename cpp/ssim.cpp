#include<iostream>
#include<torch/torch.h>


torch::Tensor fspecial_gauss(int size, float sigma){
    torch::Tensor coords = torch::arange(size).to(torch::kFloat32);
    coords -= size/2;
    torch::Tensor g = torch::exp(-(coords * coords) / (2 * sigma * sigma));
    std::cout << g << std::endl;
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

void test_gaussian(){
    std::string device = "cpu";
    if(torch::cuda::is_available()) device = "cuda";
    torch::Tensor X = torch::rand({32, 3, 128, 128}).to(device);
    torch::Tensor win = fspecial_gauss(11, 1.5).to(device);
    win = win.repeat({3, 1, 1, 1});
    std::cout << gaussian_filter_2d(X, win);
}

int main(){
    // std::cout << fspecial_gauss(3, 0.5);
    test_gaussian();
}