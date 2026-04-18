#include <torch/torch.h>
#include <iostream>
void tensor_function () {
    torch::Tensor a = torch::ones ({3, 4}); //3x4 Matrix of 1
    torch::Tensor b = torch::rand ({3, 4}); //3x4 Matrix of random values from [0, 1]
    torch::Tensor c = torch::zeros ({2, 3}, torch::kFloat64); //2x3 Matrix of zeroes, using kFloat64 definition
    //Raw Data Input
    std::vector<float> data = {1, 2, 3, 4};
    torch::Tensor t = torch::tensor (data).reshape ({2, 2});
    //Basic Operands
    torch::Tensor sum = a + b;
    torch::Tensor mm = torch::matmul (a, b.t ()); //matrix multiplication
    torch::Tensor sl = a.slice (/*dim=*/0, 0, 2); //Slicing a Tensor
    //Printout
    std::cout << t << "\n";
    std::cout << t.sizes () << "\n"; //Shape of Tensor Array
    std::cout << t.dtype () << "\n"; //Type of Tensor Array
} void autograd_function () {
    torch::Tensor x = torch::tensor ({2.0f}, torch::requires_grad ());
    torch::Tensor y = x * x * 3; //Equation of y = 3x ^ 2 in terms of tensor multiplicative
    y.backward ();
    std::cout << x.grad (); //dx / dy standard derivatives, resulting in 6x = 12
} void device_probe_function () {
    torch::Device device = torch::cuda::is_available () ? torch::kCUDA : torch::kCPU;
    torch::Tensor t = torch::rand ({3, 3}).to (device);
} void define_neural_net_function () {
    struct Net : torch::nn::Module {
        torch::nn::linear fc1 {nullptr}, fc2 {nullptr};
        net () {
            fc1 = register_module ("fc1", torch::nn::Linear (784, 128));
            fc2 = register_module ("fc2", torch::nn::Linear (128, 10));
        } torch::Tensor forward (torch::Tensor x) {
            x = torch::relu (fc1->forward (x));
            x = fc2->forward (x);
            return torch::log_softmax (x, /*dim=*/1);
        }
    };
} void training_loop_function () {
    auto net_n = std::make_shared<net> ();
    torch::optim::SGD optimizer (net_n->parameters (), /*lr=*/0.01);
    for (int epoch = 0; epoch < 10; epoch++) {
        optimizer.zero_grad ();
        torch::Tensor input = torch::rand ({32, 784}); //Fake Buffer Input
        torch::Tensor target = torch::randint (0, 10, {32}); //Target amount of training and data processed with response
        torch::Tensor output = net_n->forward (input); //Actual Output of training results
        torch::Tensor loss = torch::nll_loss (output, target); //Difference betweenm experimental and theoretical (loss in projected training)
        loss.backward ();
        optimizer.step ();
        std::cout << "Epoch: " << epoch << "Loss: " << loss.item<float> () << "\n";
    }
} void save_and_load_function () {
    auto net_n = std::make_shared<net> ();
    torch::save (net_n, "model_test.pt");
    auto net_n2 = std::make_shared<net> ();
    torch::load (net_n2, "model_test.pt");
} void load_torch_predefined_model () {
    torch::jit::script::Module module = torch::jit::load ("model_test_2.pt");
    module.to (torch::kCUDA);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back (torch::rand ({1, 3, 224, 224}).to (torch::kCUDA));
    torch::Tensor output = module.forward (inputs).toTensor ();
} int main () {
    //call functions when required
    return 0;
}
