#include <torch/torch.h>
#include <iostream>
void main_algorithm () {
    torch::Tensor a = torch::ones ({3, 4});
    torch::Tensor b = torch::rand ({3, 4});
    torch::Tensor c = torch::zeros ({2, 3}, torch::kFloat64);
    //Raw Data Input
} int main () {
    main_algorithm ();
    return 0;
}
