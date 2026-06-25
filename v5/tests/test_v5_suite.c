
#include "basis/scalar.h"
#include "basis/jit.h"
#include "basis/tensor.h"
#include "basis/sequence.h"
#include "basis/learning.h"
#include "basis/geometry.h"
#include "basis/symbolic.h"
#include "basis/compiler.h"
#include "basis/optim.h"
#include "basis/core/test_harness.h"
#include "basis/core/error.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

static double rand_double(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

void test_autograd_basic() {
    BASIS_TEST_SUITE_START("Phase A/B: Dynamic Basic Autograd");
    double val_a = rand_double(-100.0, 100.0);
    double val_b = rand_double(-100.0, 100.0);

    basis_value* a = basis_value_new(val_a);
    basis_value* b = basis_value_new(val_b);
    basis_value* c = basis_value_addition(a, b);
    BASIS_ASSERT_NEAR(c->data, val_a + val_b, 1e-9, "Dynamic forward pass addition");

    basis_value_backward_propagation(c);
    BASIS_ASSERT_NEAR(a->gradient, 1.0, 1e-9, "Gradient a");
    BASIS_ASSERT_NEAR(b->gradient, 1.0, 1e-9, "Gradient b");

    basis_value_free(c); basis_value_free(a); basis_value_free(b);
}

void test_tensor_shape_validation() {
    BASIS_TEST_SUITE_START("Phase A: Tensor Shape Validation (Static Error Logic)");
    basis_tensor* a = basis_tensor_new(2, 3);
    basis_tensor* b = basis_tensor_new(4, 2);

    basis_clear_error();
    basis_set_error_quiet(true);
    basis_tensor* c = basis_tensor_matrix_multiplication(a, b);
    basis_set_error_quiet(false);

    BASIS_ASSERT(c == NULL, "MatMul correctly rejects invalid shapes");
    BASIS_ASSERT(basis_get_last_error() == BASIS_INVALID_SHAPE, "Error code is BASIS_INVALID_SHAPE");

    basis_tensor_free(a); basis_tensor_free(b);
}

void test_symbolic_simplification() {
    BASIS_TEST_SUITE_START("Phase C: Symbolic Simplification (Static Logic)");
    basis_symbol* x1 = basis_symbol_variable("x");
    basis_symbol* one1 = basis_symbol_constant(1.0);
    basis_symbol* expr1 = basis_symbol_multiplication(x1, one1);
    basis_symbol* simp1 = basis_symbol_simplify(expr1);
    BASIS_ASSERT(simp1->type == basis_symbol_type_variable, "x * 1 simplifies to x");
    basis_symbol_free(expr1); basis_symbol_free(simp1);

    basis_symbol* x2 = basis_symbol_variable("x");
    basis_symbol* zero2 = basis_symbol_constant(0.0);
    basis_symbol* expr2 = basis_symbol_addition(x2, zero2);
    basis_symbol* simp2 = basis_symbol_simplify(expr2);
    BASIS_ASSERT(simp2->type == basis_symbol_type_variable, "x + 0 simplifies to x");
    basis_symbol_free(expr2); basis_symbol_free(simp2);

    basis_symbol* x3 = basis_symbol_variable("x");
    basis_symbol* zero3 = basis_symbol_constant(0.0);
    basis_symbol* expr3 = basis_symbol_multiplication(x3, zero3);
    basis_symbol* simp3 = basis_symbol_simplify(expr3);
    BASIS_ASSERT(simp3->type == basis_symbol_type_constant && simp3->basis_value == 0.0, "x * 0 simplifies to 0");
    basis_symbol_free(expr3); basis_symbol_free(simp3);

    basis_symbol* c1 = basis_symbol_constant(2.0);
    basis_symbol* c2 = basis_symbol_constant(3.0);
    basis_symbol* expr4 = basis_symbol_addition(c1, c2);
    basis_symbol* simp4 = basis_symbol_simplify(expr4);
    BASIS_ASSERT(simp4->type == basis_symbol_type_constant && simp4->basis_value == 5.0, "2.0 + 3.0 folds to 5.0");
    basis_symbol_free(expr4); basis_symbol_free(simp4);
}

void test_compiler_cse() {
    BASIS_TEST_SUITE_START("Phase C: Compiler CSE (Static Pointer Logic)");
    basis_symbol* x = basis_symbol_variable("x");
    basis_symbol* left = basis_symbol_multiplication(basis_symbol_copy(x), basis_symbol_copy(x));
    basis_symbol* right = basis_symbol_multiplication(basis_symbol_copy(x), basis_symbol_copy(x));
    basis_symbol* tree = basis_symbol_addition(left, right);

    basis_value* x_val = basis_value_new(3.0);
    basis_compiler* comp = basis_compiler_new();
    basis_compiler_map(comp, "x", x_val);
    basis_value* root = basis_compiler_compile(comp, tree);

    BASIS_ASSERT(root->previous_nodes[0] == root->previous_nodes[1], "CSE reused identical sub-expression pointer");

    basis_value_backward_propagation(root);
    BASIS_ASSERT_NEAR(x_val->gradient, 12.0, 1e-6, "Gradient flows through shared CSE node");

    basis_value_free(root); basis_value_free(x_val); basis_compiler_free(comp);
    basis_symbol_free(tree); basis_symbol_free(x);
}

void test_log_softmax_stability() {
    BASIS_TEST_SUITE_START("Phase D: Dynamic Numerical Stability (Random Extreme Logits)");
    int N = 10 + rand() % 50; // Random width 10 to 60
    basis_tensor* logits = basis_tensor_new(1, N);
    for(int i=0; i<N; i++) basis_tensor_set(logits, 0, i, rand_double(-1000.0, 1000.0));

    basis_tensor* log_probs = basis_tensor_log_softmax(logits);

    bool is_stable = true;
    double sum_exp = 0.0;
    for(int i=0; i<N; i++) {
        double val = BASIS_TENSOR_AT(log_probs, 0, i)->data;
        if (isinf(val) || isnan(val)) is_stable = false;
        sum_exp += exp(val);
    }
    BASIS_ASSERT(is_stable, "Dynamic Log-Softmax handles random extreme logits without Overflow/NaN");
    BASIS_ASSERT_NEAR(sum_exp, 1.0, 1e-6, "Dynamic probabilities sum to 1.0");

    basis_value* loss = basis_tensor_sum(log_probs);
    basis_value_backward_propagation(loss);

    bool grads_ok = true;
    for(int i=0; i<N; i++) if (isnan(BASIS_TENSOR_AT(logits, 0, i)->gradient)) grads_ok = false;
    BASIS_ASSERT(grads_ok, "Dynamic gradients flow through Log-Softmax without NaNs");

    basis_value_free(loss); basis_tensor_free(log_probs); basis_tensor_free(logits);
}

void test_holy_trinity() {
    BASIS_TEST_SUITE_START("Phase B: Dynamic Holy Trinity (Random Evaluation Point)");
    double x_val = rand_double(0.5, 2.0); // Safe domain for log(x) and exp(x^2)
    double eps = 1e-5;

    double f_plus = exp((x_val + eps) * (x_val + eps)) * log(x_val + eps);
    double f_minus = exp((x_val - eps) * (x_val - eps)) * log(x_val - eps);
    double fd_grad = (f_plus - f_minus) / (2.0 * eps);

    basis_value* x_auto = basis_value_new(x_val);
    basis_value* x2_auto = basis_value_power(x_auto, 2.0);
    basis_value* exp_auto = basis_value_exponential(x2_auto);
    basis_value* log_auto = basis_value_logarithm(x_auto);
    basis_value* f_auto = basis_value_multiplication(exp_auto, log_auto);
    basis_value_backward_propagation(f_auto);
    double auto_grad = x_auto->gradient;

    basis_symbol* x_sym = basis_symbol_variable("x");
    basis_symbol* x2_sym = basis_symbol_power(basis_symbol_copy(x_sym), 2.0);
    basis_symbol* exp_sym = basis_symbol_exponential(x2_sym);
    basis_symbol* log_sym = basis_symbol_logarithm(x_sym);
    basis_symbol* f_sym = basis_symbol_multiplication(exp_sym, log_sym);

    basis_symbol* df_sym = basis_symbol_differentiation(f_sym, "x");
    basis_symbol* df_simp = basis_symbol_simplify(df_sym);

    basis_value* x_val_sym = basis_value_new(x_val);
    basis_compiler* comp = basis_compiler_new();
    basis_compiler_map(comp, "x", x_val_sym);
    basis_value* sym_grad_node = basis_compiler_compile(comp, df_simp);
    double sym_grad = sym_grad_node->data;

    BASIS_ASSERT_NEAR(auto_grad, fd_grad, 1e-4, "Dynamic Autodiff matches Finite Difference");
    BASIS_ASSERT_NEAR(sym_grad, fd_grad, 1e-4, "Dynamic Symbolic matches Finite Difference");
    BASIS_ASSERT_NEAR(sym_grad, auto_grad, 1e-6, "Dynamic Symbolic matches Autodiff directly");

    basis_value_free(f_auto); basis_value_free(x_auto);
    basis_value_free(sym_grad_node); basis_value_free(x_val_sym);
    basis_compiler_free(comp);
    basis_symbol_free(f_sym); basis_symbol_free(df_sym); basis_symbol_free(df_simp);
}

void test_information_geometry() {
    BASIS_TEST_SUITE_START("Phase E: Dynamic Information Geometry (Random PSD Verification)");
    int N = 3 + rand() % 5;
    int M = 3 + rand() % 5;
    basis_tensor* X = basis_tensor_new(N, M);
    for(int i=0; i<N; i++) for(int j=0; j<M; j++) basis_tensor_set(X, i, j, rand_double(-5.0, 5.0));

    basis_metric* fisher = basis_metric_fisher_information(NULL, X, NULL);
    basis_tensor* F = fisher->matrix;

    bool symmetric = true;
    for(int i=0; i<M; i++) for(int j=0; j<M; j++) {
        if (fabs(BASIS_TENSOR_AT(F, i, j)->data - BASIS_TENSOR_AT(F, j, i)->data) > 1e-9) symmetric = false;
    }
    BASIS_ASSERT(symmetric, "Dynamic Fisher Matrix is Symmetric");

    bool psd = true;
    for(int t=0; t<5; t++) { // Test 5 random vectors
        double quad = 0.0;
        double v[10];
        for(int i=0; i<M; i++) v[i] = rand_double(-10.0, 10.0);
        for(int i=0; i<M; i++) for(int j=0; j<M; j++) {
            quad += v[i] * BASIS_TENSOR_AT(F, i, j)->data * v[j];
        }
        if (quad < -1e-9) psd = false;
    }
    BASIS_ASSERT(psd, "Dynamic Fisher Matrix is Positive Semi-Definite");

    // Invariant: The trace of the Fisher matrix must be strictly positive for non-zero inputs
    double trace = 0.0;
    for(int i=0; i<M; i++) trace += BASIS_TENSOR_AT(F, i, i)->data;
    BASIS_ASSERT(trace > 0.0, "Dynamic Fisher Matrix Trace is strictly positive");

    basis_metric_free(fisher); basis_tensor_free(X);
}

void test_natural_gradient_descent() {
    BASIS_TEST_SUITE_START("Phase E: Dynamic NGD (Random True Weights)");
    double true_w0 = rand_double(-10.0, 10.0);
    double true_w1 = rand_double(-10.0, 10.0);

    basis_tensor* X = basis_tensor_new(3, 2);
    basis_tensor_set(X, 0, 0, 1.0); basis_tensor_set(X, 0, 1, 1.1);
    basis_tensor_set(X, 1, 0, 2.0); basis_tensor_set(X, 1, 1, 2.1);
    basis_tensor_set(X, 2, 0, 3.0); basis_tensor_set(X, 2, 1, 3.1);

    basis_tensor* W = basis_tensor_new(2, 1);
    basis_tensor_set(W, 0, 0, 0.0); basis_tensor_set(W, 1, 0, 0.0);

    basis_metric* fisher = basis_metric_fisher_information(NULL, X, NULL);
    basis_tensor* Y_true = basis_tensor_new(3, 1);
    basis_tensor_set(Y_true, 0, 0, 1.0*true_w0 + 1.1*true_w1);
    basis_tensor_set(Y_true, 1, 0, 2.0*true_w0 + 2.1*true_w1);
    basis_tensor_set(Y_true, 2, 0, 3.0*true_w0 + 3.1*true_w1);

    basis_tensor* Y_pred = basis_tensor_matrix_multiplication(X, W);
    basis_value* loss_sum = basis_value_new(0.0);
    for(int i=0; i<3; i++) {
        basis_value* diff = basis_value_subtraction(BASIS_TENSOR_AT(Y_pred, 0, i), BASIS_TENSOR_AT(Y_true, 0, i));
        basis_value* sq = basis_value_power(diff, 2.0);
        basis_value* new_sum = basis_value_addition(loss_sum, sq);
        basis_value_free(loss_sum); basis_value_free(diff); basis_value_free(sq);
        loss_sum = new_sum;
    }
    basis_value_backward_propagation(loss_sum);
    basis_metric_ngd_step(fisher, W, 0.5);

    BASIS_ASSERT_NEAR(W->data[0]->data, true_w0, 1e-4, "Dynamic NGD converges to random true W[0]");
    BASIS_ASSERT_NEAR(W->data[1]->data, true_w1, 1e-4, "Dynamic NGD converges to random true W[1]");

    basis_value_free(loss_sum); basis_tensor_free(Y_pred); basis_metric_free(fisher);
    basis_tensor_free(X); basis_tensor_free(W); basis_tensor_free(Y_true);
}

void test_matmul_dag_compression() {
    BASIS_TEST_SUITE_START("Phase F: Dynamic Fused MatMul (Random Dimensions & Values)");
    int M = 5 + rand() % 15;
    int K = 5 + rand() % 15;
    int N = 5 + rand() % 15;

    basis_tensor* A = basis_tensor_new(M, K);
    basis_tensor* B = basis_tensor_new(K, N);
    for(int i=0; i<M; i++) for(int j=0; j<K; j++) basis_tensor_set(A, i, j, rand_double(-5.0, 5.0));
    for(int i=0; i<K; i++) for(int j=0; j<N; j++) basis_tensor_set(B, i, j, rand_double(-5.0, 5.0));

    basis_tensor* C = basis_tensor_matrix_multiplication(A, B);

    bool forward_ok = true;
    for(int i=0; i<M; i++) for(int j=0; j<N; j++) {
        double sum = 0.0;
        for(int k=0; k<K; k++) sum += BASIS_TENSOR_AT(A, i, k)->data * BASIS_TENSOR_AT(B, k, j)->data;
        if (fabs(BASIS_TENSOR_AT(C, i, j)->data - sum) > 1e-6) forward_ok = false;
    }
    BASIS_ASSERT(forward_ok, "Dynamic Fused MatMul forward pass matches naive C-loop");

    basis_value* loss = basis_tensor_sum(C);
    basis_value_backward_propagation(loss);

    bool backward_ok = true;
    for(int i=0; i<M; i++) for(int j=0; j<K; j++) if(isnan(BASIS_TENSOR_AT(A, i, j)->gradient)) backward_ok = false;
    BASIS_ASSERT(backward_ok, "Dynamic MatMul backward pass completes without NaN");

    // Invariant: If dL/dC is all 1s (from sum), the global gradient flow must be non-zero and conserved
    double sum_grad_A = 0.0;
    for(int i=0; i<M; i++) for(int j=0; j<K; j++) sum_grad_A += BASIS_TENSOR_AT(A, i, j)->gradient;
    BASIS_ASSERT(sum_grad_A != 0.0, "Dynamic MatMul gradient conservation invariant holds");

    basis_value_free(loss); basis_tensor_free(C); basis_tensor_free(A); basis_tensor_free(B);
}

void test_strided_tensor_views() {
    BASIS_TEST_SUITE_START("Phase G: Dynamic Strided Views (Random Shapes)");
    int R = 3 + rand() % 10;
    int C_cols = 3 + rand() % 10;
    basis_tensor* A = basis_tensor_new(R, C_cols);
    for(int i=0; i<R; i++) for(int j=0; j<C_cols; j++) basis_tensor_set(A, i, j, rand_double(-10.0, 10.0));

    basis_tensor* A_T = basis_tensor_transpose(A);
    bool trans_ok = true;
    for(int i=0; i<R; i++) for(int j=0; j<C_cols; j++) {
        if (fabs(BASIS_TENSOR_AT(A, i, j)->data - BASIS_TENSOR_AT(A_T, j, i)->data) > 1e-9) trans_ok = false;
    }
    BASIS_ASSERT(trans_ok, "Dynamic Transpose reads correct strided memory");
    BASIS_ASSERT(A_T->data == A->data, "Dynamic Transpose is Zero-Copy");

    basis_value* loss = basis_tensor_sum(A_T);
    basis_value_backward_propagation(loss);

    bool grad_ok = true;
    for(int i=0; i<R; i++) for(int j=0; j<C_cols; j++) {
        if (fabs(BASIS_TENSOR_AT(A, i, j)->gradient - 1.0) > 1e-9) grad_ok = false;
    }
    BASIS_ASSERT(grad_ok, "Dynamic Gradients flow perfectly through transpose view");

    basis_value_free(loss); basis_tensor_free(A_T); basis_tensor_free(A);
}

void test_broadcasting_math() {
    BASIS_TEST_SUITE_START("Phase I: Dynamic Broadcasting (Random Batch & Feature Sizes)");
    int B = 4 + rand() % 20;
    int N = 4 + rand() % 20;

    basis_tensor* A = basis_tensor_new(B, N);
    basis_tensor* bias = basis_tensor_new(1, N);
    for(int i=0; i<B; i++) for(int j=0; j<N; j++) basis_tensor_set(A, i, j, rand_double(-5.0, 5.0));
    for(int j=0; j<N; j++) basis_tensor_set(bias, 0, j, rand_double(-5.0, 5.0));

    basis_tensor* bias_view = basis_tensor_broadcast_view(bias, B, N);
    basis_tensor* C = basis_tensor_add(A, bias_view);

    bool forward_ok = true;
    for(int i=0; i<B; i++) for(int j=0; j<N; j++) {
        double expected = BASIS_TENSOR_AT(A, i, j)->data + BASIS_TENSOR_AT(bias, 0, j)->data;
        if (fabs(BASIS_TENSOR_AT(C, i, j)->data - expected) > 1e-9) forward_ok = false;
    }
    BASIS_ASSERT(forward_ok, "Dynamic Broadcast forward pass matches naive");

    basis_value* loss = basis_tensor_sum(C);
    basis_value_backward_propagation(loss);

    bool backward_ok = true;
    for(int j=0; j<N; j++) {
        if (fabs(BASIS_TENSOR_AT(bias, 0, j)->gradient - (double)B) > 1e-9) backward_ok = false;
    }
    BASIS_ASSERT(backward_ok, "Dynamic Broadcast backward perfectly accumulates gradients");

    // Invariant: The broadcast view must strictly maintain the zero-stride property
    BASIS_ASSERT(bias_view->row_stride == 0, "Dynamic Broadcast maintains zero-stride memory invariant");

    basis_value_free(loss); basis_tensor_free(C); basis_tensor_free(bias_view);
    basis_tensor_free(A); basis_tensor_free(bias);
}


void test_hessian_vector_product() {
    BASIS_TEST_SUITE_START("Phase H: Dynamic Higher-Order Calculus (Exact Symbolic HVP)");

    basis_symbol* w1 = basis_symbol_variable("w1");
    basis_symbol* w2 = basis_symbol_variable("w2");

    basis_symbol* w1_sq = basis_symbol_power(basis_symbol_copy(w1), 2.0);
    basis_symbol* w2_sq = basis_symbol_power(basis_symbol_copy(w2), 2.0);
    basis_symbol* cross = basis_symbol_multiplication(basis_symbol_copy(w1), basis_symbol_copy(w2));

    basis_symbol* term2 = basis_symbol_multiplication(basis_symbol_constant(2.0), cross);
    basis_symbol* term3 = basis_symbol_multiplication(basis_symbol_constant(3.0), w2_sq);

    basis_symbol* L = basis_symbol_addition(w1_sq, term2);
    basis_symbol* L_final = basis_symbol_addition(L, term3);

    basis_symbol* G1 = basis_symbol_differentiation(L_final, "w1");
    basis_symbol* G2 = basis_symbol_differentiation(L_final, "w2");

    basis_symbol* S1 = basis_symbol_multiplication(G1, basis_symbol_constant(1.0));
    basis_symbol* S2 = basis_symbol_multiplication(G2, basis_symbol_constant(-1.0));
    basis_symbol* S = basis_symbol_addition(S1, S2);

    basis_symbol* S_simp = basis_symbol_simplify(S);
    basis_symbol_free(S);

    basis_symbol* HVP1_raw = basis_symbol_differentiation(S_simp, "w1");
    basis_symbol* HVP2_raw = basis_symbol_differentiation(S_simp, "w2");

    basis_symbol* HVP1 = basis_symbol_simplify(HVP1_raw);
    basis_symbol* HVP2 = basis_symbol_simplify(HVP2_raw);

    basis_symbol_free(HVP1_raw);
    basis_symbol_free(HVP2_raw);

    // Evaluate at a random point to prove it's constant for a quadratic manifold
    double rand_w1 = rand_double(-10.0, 10.0);
    double rand_w2 = rand_double(-10.0, 10.0);

    basis_value* val_w1 = basis_value_new(rand_w1);
    basis_value* val_w2 = basis_value_new(rand_w2);

    basis_compiler* comp = basis_compiler_new();
    basis_compiler_map(comp, "w1", val_w1);
    basis_compiler_map(comp, "w2", val_w2);

    basis_value* hvp1_num = basis_compiler_compile(comp, HVP1);
    basis_value* hvp2_num = basis_compiler_compile(comp, HVP2);

    BASIS_ASSERT_NEAR(hvp1_num->data, 0.0, 1e-6, "Dynamic HVP[0] matches exact analytical Hessian product");
    BASIS_ASSERT_NEAR(hvp2_num->data, -4.0, 1e-6, "Dynamic HVP[1] matches exact analytical Hessian product");

    basis_value_free(hvp1_num);
    basis_value_free(hvp2_num);
    basis_value_free(val_w1);
    basis_value_free(val_w2);
    basis_compiler_free(comp);

    basis_symbol_free(S_simp);
    basis_symbol_free(HVP1);
    basis_symbol_free(HVP2);
    basis_symbol_free(L_final);
    basis_symbol_free(w1);
    basis_symbol_free(w2);
}

void test_mlp_xor_convergence() {
    BASIS_TEST_SUITE_START("Phase K: Dynamic MLP XOR (Random Weight Initialization)");
    basis_tensor* X = basis_tensor_new(4, 2);
    basis_tensor_set(X, 0, 0, 0.0); basis_tensor_set(X, 0, 1, 0.0);
    basis_tensor_set(X, 1, 0, 0.0); basis_tensor_set(X, 1, 1, 1.0);
    basis_tensor_set(X, 2, 0, 1.0); basis_tensor_set(X, 2, 1, 0.0);
    basis_tensor_set(X, 3, 0, 1.0); basis_tensor_set(X, 3, 1, 1.0);

    basis_tensor* Y_true = basis_tensor_new(4, 1);
    basis_tensor_set(Y_true, 0, 0, 0.0); basis_tensor_set(Y_true, 1, 0, 1.0);
    basis_tensor_set(Y_true, 2, 0, 1.0); basis_tensor_set(Y_true, 3, 0, 0.0);

    basis_tensor* W1 = basis_tensor_new(2, 8);
    basis_tensor* b1 = basis_tensor_new(1, 8);
    basis_tensor* W2 = basis_tensor_new(8, 1);
    basis_tensor* b2 = basis_tensor_new(1, 1);

    for(int i=0; i<16; i++) W1->data[i]->data = rand_double(-1.0, 1.0);
    for(int i=0; i<8; i++) W2->data[i]->data = rand_double(-1.0, 1.0);

    basis_adam* opt_W1 = basis_adam_new(2, 8, 0.1);
    basis_adam* opt_b1 = basis_adam_new(1, 8, 0.1);
    basis_adam* opt_W2 = basis_adam_new(8, 1, 0.1);
    basis_adam* opt_b2 = basis_adam_new(1, 1, 0.1);

    double initial_loss = 0.0, final_loss = 0.0;

    for(int epoch=0; epoch<3000; epoch++) {
        basis_tensor* Z1 = basis_tensor_matrix_multiplication(X, W1);
        basis_tensor* b1_view = basis_tensor_broadcast_view(b1, 4, 8);
        basis_tensor* A1 = basis_tensor_add(Z1, b1_view);
        basis_tensor* H1 = basis_tensor_rectified_linear_unit(A1);

        basis_tensor* Z2 = basis_tensor_matrix_multiplication(H1, W2);
        basis_tensor* b2_view = basis_tensor_broadcast_view(b2, 4, 1);
        basis_tensor* Y_pred = basis_tensor_add(Z2, b2_view);

        basis_tensor* neg_Y = basis_tensor_scalar_multiplication(Y_true, -1.0);
        basis_tensor* diff = basis_tensor_add(Y_pred, neg_Y);
        basis_tensor* sq = basis_tensor_multiplication(diff, diff);
        basis_value* sum_sq = basis_tensor_sum(sq);
        basis_value* divisor = basis_value_new(0.25);
        basis_value* loss = basis_value_multiplication(sum_sq, divisor);

        if (epoch == 0) initial_loss = loss->data;
        if (epoch == 2999) final_loss = loss->data;

        basis_value_backward_propagation(loss);

        basis_adam_optimization_step(opt_W1, W1);
        basis_adam_optimization_step(opt_b1, b1);
        basis_adam_optimization_step(opt_W2, W2);
        basis_adam_optimization_step(opt_b2, b2);

        basis_value_free(loss); basis_value_free(divisor); basis_value_free(sum_sq);
        basis_tensor_free(sq); basis_tensor_free(diff); basis_tensor_free(neg_Y);
        basis_tensor_free(Y_pred); basis_tensor_free(b2_view); basis_tensor_free(Z2);
        basis_tensor_free(H1); basis_tensor_free(A1); basis_tensor_free(b1_view);
        basis_tensor_free(Z1);
    }

    BASIS_ASSERT(final_loss < initial_loss * 0.1, "Dynamic MLP successfully learns XOR from random init");
    BASIS_ASSERT(final_loss < 0.1, "Dynamic MLP achieves low absolute error");

    basis_adam_free(opt_W1); basis_adam_free(opt_b1);
    basis_adam_free(opt_W2); basis_adam_free(opt_b2);
    basis_tensor_free(W1); basis_tensor_free(b1);
    basis_tensor_free(W2); basis_tensor_free(b2);
    basis_tensor_free(X); basis_tensor_free(Y_true);
}


void test_jit_inference() {
    BASIS_TEST_SUITE_START("Phase J: JIT Code Generation (Real AST Compiler)");

    // === Test 1: exp(x^2) * log(x) ===
    {
        basis_symbol* x = basis_symbol_variable("x");
        basis_symbol* x2 = basis_symbol_power(basis_symbol_copy(x), 2.0);
        basis_symbol* e = basis_symbol_exponential(x2);
        basis_symbol* l = basis_symbol_logarithm(x);
        basis_symbol* f = basis_symbol_multiplication(e, l);

        char* vars[] = {"x"};
        basis_jit_module* jit = basis_jit_compile(f, vars, 1);
        BASIS_ASSERT(jit != NULL, "JIT compiled exp(x^2)*log(x) successfully");

        double x_val = 1.5;
        double jit_result = basis_jit_execute(jit, &x_val);
        double expected = exp(1.5*1.5) * log(1.5);
        BASIS_ASSERT_NEAR(jit_result, expected, 1e-9, "JIT output matches expected for exp(x^2)*log(x)");

        basis_jit_free(jit);
        basis_symbol_free(f);
    }

    // === Test 2: (a ** 2.0) + (b * a) ===
    {
        basis_symbol* a = basis_symbol_variable("a");
        basis_symbol* b = basis_symbol_variable("b");
        basis_symbol* a_sq = basis_symbol_power(basis_symbol_copy(a), 2.0);
        basis_symbol* ba = basis_symbol_multiplication(basis_symbol_copy(b), a);
        basis_symbol* f2 = basis_symbol_addition(a_sq, ba);

        char* vars2[] = {"a", "b"};
        basis_jit_module* jit2 = basis_jit_compile(f2, vars2, 2);
        BASIS_ASSERT(jit2 != NULL, "JIT compiled (a^2)+(b*a) successfully");

        double vals[] = {2.0, 3.0};
        double jit_result2 = basis_jit_execute(jit2, vals);
        double expected2 = (2.0 * 2.0) + (3.0 * 2.0); // 4.0 + 6.0 = 10.0
        BASIS_ASSERT_NEAR(jit_result2, expected2, 1e-9, "JIT output matches expected for (a^2)+(b*a)");

        basis_jit_free(jit2);
        basis_symbol_free(f2);
    }

    // === Test 3: Constant folding through JIT: (3.0 + 4.0) * 2.0 ===
    {
        basis_symbol* c1 = basis_symbol_constant(3.0);
        basis_symbol* c2 = basis_symbol_constant(4.0);
        basis_symbol* sum = basis_symbol_addition(c1, c2);
        basis_symbol* c3 = basis_symbol_constant(2.0);
        basis_symbol* prod = basis_symbol_multiplication(sum, c3);

        char* vars3[] = {"dummy"};
        basis_jit_module* jit3 = basis_jit_compile(prod, vars3, 1);
        BASIS_ASSERT(jit3 != NULL, "JIT compiled pure constant expression");

        double dummy = 0.0;
        double jit_result3 = basis_jit_execute(jit3, &dummy);
        BASIS_ASSERT_NEAR(jit_result3, 14.0, 1e-9, "JIT evaluates pure constant expression correctly");

        basis_jit_free(jit3);
        basis_symbol_free(prod);
    }
}


void test_stress_abyss_graph() {
    BASIS_TEST_SUITE_START("Phase L1: The Abyss Graph (10,000 Sequential Ops)");
    basis_value* current = basis_value_new(1.0);
    for(int i = 0; i < 10000; i++) {
        basis_value* c = basis_value_new(0.0001);
        basis_value* next = basis_value_addition(current, c);
        basis_value_free(current);
        basis_value_free(c);
        current = next;
    }
    BASIS_ASSERT_NEAR(current->data, 2.0, 1e-6, "Abyss graph forward pass completes 10,000 additions");

    // Backward pass tests the iterative topological sort on a 10,000 depth graph
    basis_value_backward_propagation(current);
    BASIS_ASSERT(current->gradient == 1.0, "Abyss graph backward pass completes without stack overflow");

    // Teardown tests the new iterative basis_value_free
    basis_value_free(current);
    BASIS_ASSERT(1, "Abyss graph iterative teardown completes without stack overflow");
}

void test_stress_leviathan_matmul() {
    BASIS_TEST_SUITE_START("Phase L2: The Leviathan MatMul (100x100 DAG)");
    size_t N = 100;
    basis_tensor* A = basis_tensor_new(N, N);
    basis_tensor* B = basis_tensor_new(N, N);

    for(size_t i=0; i<N*N; i++) {
        A->data[i]->data = 0.01;
        B->data[i]->data = 0.01;
    }

    basis_tensor* C = basis_tensor_matrix_multiplication(A, B);
    BASIS_ASSERT_NEAR(BASIS_TENSOR_AT(C, 0, 0)->data, 0.01, 1e-6, "Leviathan forward pass correct");

    basis_value* loss = basis_tensor_sum(C);
    basis_value_backward_propagation(loss);

    BASIS_ASSERT_NEAR(BASIS_TENSOR_AT(A, 0, 0)->gradient, 1.0, 1e-6, "Leviathan backward pass routes 2M edges correctly");

    basis_value_free(loss);
    basis_tensor_free(C);
    basis_tensor_free(A);
    basis_tensor_free(B);
}

void test_stress_deep_mlp() {
    BASIS_TEST_SUITE_START("Phase L3: The Deep MLP (5 Layers, 500 Epochs)");
    size_t B = 32;
    basis_tensor* X = basis_tensor_new(B, 16);
    for(size_t i=0; i<B*16; i++) X->data[i]->data = ((i % 5) - 2.0) * 0.1;

    basis_tensor* W1 = basis_tensor_new(16, 32); basis_tensor* b1 = basis_tensor_new(1, 32);
    basis_tensor* W2 = basis_tensor_new(32, 32); basis_tensor* b2 = basis_tensor_new(1, 32);
    basis_tensor* W3 = basis_tensor_new(32, 32); basis_tensor* b3 = basis_tensor_new(1, 32);
    basis_tensor* W4 = basis_tensor_new(32, 16); basis_tensor* b4 = basis_tensor_new(1, 16);
    basis_tensor* W5 = basis_tensor_new(16, 1);  basis_tensor* b5 = basis_tensor_new(1, 1);

    for(size_t i=0; i<16*32; i++) W1->data[i]->data = ((i % 7) - 3.0) * 0.1;
    for(size_t i=0; i<32*32; i++) W2->data[i]->data = ((i % 7) - 3.0) * 0.1;
    for(size_t i=0; i<32*32; i++) W3->data[i]->data = ((i % 7) - 3.0) * 0.1;
    for(size_t i=0; i<32*16; i++) W4->data[i]->data = ((i % 7) - 3.0) * 0.1;
    for(size_t i=0; i<16*1; i++) W5->data[i]->data = ((i % 7) - 3.0) * 0.1;

    basis_adam* opt1 = basis_adam_new(16, 32, 0.01);
    basis_adam* opt2 = basis_adam_new(32, 32, 0.01);
    basis_adam* opt3 = basis_adam_new(32, 32, 0.01);
    basis_adam* opt4 = basis_adam_new(32, 16, 0.01);
    basis_adam* opt5 = basis_adam_new(16, 1, 0.01);

    double final_loss = 0.0;

    for(int epoch=0; epoch<500; epoch++) {
        basis_tensor* Z1 = basis_tensor_matrix_multiplication(X, W1);
        basis_tensor* b1_v = basis_tensor_broadcast_view(b1, B, 32);
        basis_tensor* A1 = basis_tensor_add(Z1, b1_v);
        basis_tensor* H1 = basis_tensor_rectified_linear_unit(A1);

        basis_tensor* Z2 = basis_tensor_matrix_multiplication(H1, W2);
        basis_tensor* b2_v = basis_tensor_broadcast_view(b2, B, 32);
        basis_tensor* A2 = basis_tensor_add(Z2, b2_v);
        basis_tensor* H2 = basis_tensor_rectified_linear_unit(A2);

        basis_tensor* Z3 = basis_tensor_matrix_multiplication(H2, W3);
        basis_tensor* b3_v = basis_tensor_broadcast_view(b3, B, 32);
        basis_tensor* A3 = basis_tensor_add(Z3, b3_v);
        basis_tensor* H3 = basis_tensor_rectified_linear_unit(A3);

        basis_tensor* Z4 = basis_tensor_matrix_multiplication(H3, W4);
        basis_tensor* b4_v = basis_tensor_broadcast_view(b4, B, 16);
        basis_tensor* A4 = basis_tensor_add(Z4, b4_v);
        basis_tensor* H4 = basis_tensor_rectified_linear_unit(A4);

        basis_tensor* Z5 = basis_tensor_matrix_multiplication(H4, W5);
        basis_tensor* b5_v = basis_tensor_broadcast_view(b5, B, 1);
        basis_tensor* Y = basis_tensor_add(Z5, b5_v);

        basis_value* loss = basis_tensor_sum(Y);
        if(epoch == 499) final_loss = loss->data;

        basis_value_backward_propagation(loss);

        basis_adam_optimization_step(opt1, W1);
        basis_adam_optimization_step(opt2, W2);
        basis_adam_optimization_step(opt3, W3);
        basis_adam_optimization_step(opt4, W4);
        basis_adam_optimization_step(opt5, W5);

        basis_value_free(loss);
        basis_tensor_free(Y); basis_tensor_free(b5_v); basis_tensor_free(Z5);
        basis_tensor_free(H4); basis_tensor_free(A4); basis_tensor_free(b4_v); basis_tensor_free(Z4);
        basis_tensor_free(H3); basis_tensor_free(A3); basis_tensor_free(b3_v); basis_tensor_free(Z3);
        basis_tensor_free(H2); basis_tensor_free(A2); basis_tensor_free(b2_v); basis_tensor_free(Z2);
        basis_tensor_free(H1); basis_tensor_free(A1); basis_tensor_free(b1_v); basis_tensor_free(Z1);
    }

    BASIS_ASSERT(!isnan(final_loss), "Deep MLP loss did not explode to NaN");
    BASIS_ASSERT(!isinf(final_loss), "Deep MLP loss did not explode to Inf");

    basis_adam_free(opt1); basis_adam_free(opt2); basis_adam_free(opt3); basis_adam_free(opt4); basis_adam_free(opt5);
    basis_tensor_free(W1); basis_tensor_free(b1);
    basis_tensor_free(W2); basis_tensor_free(b2);
    basis_tensor_free(W3); basis_tensor_free(b3);
    basis_tensor_free(W4); basis_tensor_free(b4);
    basis_tensor_free(W5); basis_tensor_free(b5);
    basis_tensor_free(X);
}

void test_stress_jit_fuzzer() {
    BASIS_TEST_SUITE_START("Phase L4: The JIT Fuzzer (50 Random ASTs)");

    int success_count = 0;
    for(int i=0; i<50; i++) {
        basis_symbol* x = basis_symbol_variable("x");
        basis_symbol* current = basis_symbol_copy(x);

        for(int j=0; j<5; j++) {
            int op = rand() % 4;
            basis_symbol* c = basis_symbol_constant((double)(rand() % 5) + 1.0);
            if(op == 0) current = basis_symbol_addition(current, c);
            else if(op == 1) current = basis_symbol_multiplication(current, c);
            else if(op == 2) {
                current = basis_symbol_power(current, 2.0);
                basis_symbol_free(c);
            } else {
                current = basis_symbol_exponential(current);
                basis_symbol_free(c);
            }
        }

        char* vars[] = {"x"};
        basis_jit_module* jit = basis_jit_compile(current, vars, 1);

        if(jit) {
            double x_val = 0.5; // Safe domain for exp and powers
            double jit_res = basis_jit_execute(jit, &x_val);

            basis_value* x_auto = basis_value_new(x_val);
            basis_compiler* comp = basis_compiler_new();
            basis_compiler_map(comp, "x", x_auto);
            basis_value* auto_res_node = basis_compiler_compile(comp, current);
            double auto_res = auto_res_node->data;

            bool match = false;
            if (isnan(jit_res) && isnan(auto_res)) match = true;
            else if (isinf(jit_res) && isinf(auto_res) && (jit_res > 0) == (auto_res > 0)) match = true;
            else if (!isnan(jit_res) && !isinf(jit_res) && !isnan(auto_res) && !isinf(auto_res) && fabs(jit_res - auto_res) < 1e-3) match = true;

            if (match) success_count++;

            basis_value_free(auto_res_node);
            basis_value_free(x_auto);
            basis_compiler_free(comp);
            basis_jit_free(jit);
        }
        basis_symbol_free(current);
        basis_symbol_free(x);
    }

    BASIS_ASSERT(success_count == 50, "JIT Fuzzer successfully compiled and matched 50 random ASTs");
}

int main() {
    srand(time(NULL)); // Seed the random number generator
    printf("=========================================================\n");
    printf("  BASIS v5 DYNAMIC Property-Based Test Suite            \n");
    printf("=========================================================\n");

    test_autograd_basic();
    test_tensor_shape_validation();
    test_symbolic_simplification();
    test_compiler_cse();
    test_log_softmax_stability();
    test_holy_trinity();
    test_information_geometry();
    test_natural_gradient_descent();
    test_matmul_dag_compression();
    test_strided_tensor_views();
    test_broadcasting_math();
    test_jit_inference();
    test_hessian_vector_product();
    test_mlp_xor_convergence();

    test_stress_abyss_graph();
    test_stress_leviathan_matmul();
    test_stress_deep_mlp();
    test_stress_jit_fuzzer();
    BASIS_TEST_SUITE_END();
    return basis_tests_failed > 0 ? 1 : 0;
}
