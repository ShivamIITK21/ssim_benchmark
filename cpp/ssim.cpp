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

// torch::Tensor gaussian_filter(torch::Tensor inp, torch::Tensor win){
//     auto shape = inp.sizes();
//     int dimsum = 0;
//     for(auto dim : shape) dimsum += dim;

//     torch::nn::Conv2d conv2;
//     torch::nn::Conv3d conv3;

//     int c = shape[1];
//     torch::Tensor out = inp.detach().clone();

// }

int main(){
    std::cout << fspecial_gauss(3, 0.5);
}