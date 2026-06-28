#include <basis.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PI 3.14159265358979323846
#define NUM_COLLOC 30
#define NUM_IC 15
#define EPOCHS 1000
#define EPS 1e-4

// Evaluates the MLP and applies the hard-constrained BC Ansatz: u(0,t)=0, u(1,t)=0
basis_tensor* eval_u(double x_val, double t_val, basis_tensor* W1, basis_tensor* b1, basis_tensor* W2, basis_tensor* b2) {
    basis_tensor* X_in = basis_tensor_new(1, 2);
    basis_tensor_set(X_in, 0, 0, x_val); basis_tensor_set(X_in, 0, 1, t_val);

    basis_tensor* Z1 = basis_tensor_matrix_multiplication(X_in, W1);
    basis_tensor* b1_v = basis_tensor_broadcast_view(b1, 1, 16);
    basis_tensor* A1 = basis_tensor_add(Z1, b1_v);
    basis_tensor* H1 = basis_tensor_tanh(A1); // Smooth activation for 2nd-order derivatives

    basis_tensor* Z2 = basis_tensor_matrix_multiplication(H1, W2);
    basis_tensor* b2_v = basis_tensor_broadcast_view(b2, 1, 1);
    basis_tensor* U_raw = basis_tensor_add(Z2, b2_v);

    // Ansatz Mask: x * (1 - x)
    double mask = x_val * (1.0 - x_val);
    basis_tensor* U_masked = basis_tensor_scalar_multiplication(U_raw, mask);

    basis_tensor_free(X_in); basis_tensor_free(Z1); basis_tensor_free(b1_v);
    basis_tensor_free(A1); basis_tensor_free(H1); basis_tensor_free(Z2);
    basis_tensor_free(b2_v); basis_tensor_free(U_raw);

    return U_masked;
}

int main() {
    srand(42);
    printf("=== BASIS v5: Physics-Informed Neural Network (1D Heat Equation) ===\n");
    printf("Solving: du/dt = d^2u/dx^2\n");
    printf("BCs: u(0,t)=0, u(1,t)=0 | IC: u(x,0)=sin(pi*x)\n\n");

    basis_tensor* W1 = basis_tensor_new(2, 16);
    basis_tensor* b1 = basis_tensor_new(1, 16);
    basis_tensor* W2 = basis_tensor_new(16, 1);
    basis_tensor* b2 = basis_tensor_new(1, 1);

    for(size_t i=0; i<32; i++) W1->data[i]->data = ((rand() % 100) / 100.0 - 0.5) * 1.0;
    for(size_t i=0; i<16; i++) W2->data[i]->data = ((rand() % 100) / 100.0 - 0.5) * 1.0;

    basis_adam* opt_W1 = basis_adam_new(2, 16, 0.05);
    basis_adam* opt_b1 = basis_adam_new(1, 16, 0.05);
    basis_adam* opt_W2 = basis_adam_new(16, 1, 0.05);
    basis_adam* opt_b2 = basis_adam_new(1, 1, 0.05);

    double x_f[NUM_COLLOC], t_f[NUM_COLLOC];
    for(int i=0; i<NUM_COLLOC; i++) {
        x_f[i] = (double)rand() / RAND_MAX;
        t_f[i] = (double)rand() / RAND_MAX;
    }
    double x_i[NUM_IC];
    for(int i=0; i<NUM_IC; i++) x_i[i] = (double)rand() / RAND_MAX;

    for(int epoch=0; epoch<EPOCHS; epoch++) {
        basis_value* loss_sum = basis_value_new(0.0);

        // 1. PDE Residual Loss (Finite Difference Stencil)
        for(int i=0; i<NUM_COLLOC; i++) {
            double xc = x_f[i], tc = t_f[i];
            basis_tensor* u_c = eval_u(xc, tc, W1, b1, W2, b2);
            basis_tensor* u_tp = eval_u(xc, tc + EPS, W1, b1, W2, b2);
            basis_tensor* u_tm = eval_u(xc, tc - EPS, W1, b1, W2, b2);
            basis_tensor* u_xp = eval_u(xc + EPS, tc, W1, b1, W2, b2);
            basis_tensor* u_xm = eval_u(xc - EPS, tc, W1, b1, W2, b2);

            basis_value* v_uc = u_c->data[0]; basis_value_retain(v_uc);
            basis_value* v_utp = u_tp->data[0]; basis_value_retain(v_utp);
            basis_value* v_utm = u_tm->data[0]; basis_value_retain(v_utm);
            basis_value* v_uxp = u_xp->data[0]; basis_value_retain(v_uxp);
            basis_value* v_uxm = u_xm->data[0]; basis_value_retain(v_uxm);

            basis_tensor_free(u_c); basis_tensor_free(u_tp); basis_tensor_free(u_tm);
            basis_tensor_free(u_xp); basis_tensor_free(u_xm);

            basis_value* dt_u = basis_value_subtraction(v_utp, v_utm);
            basis_value* dt_scale = basis_value_new(1.0 / (2.0 * EPS));
            basis_value* dt_final = basis_value_multiplication(dt_u, dt_scale);

            basis_value* dxx_1 = basis_value_addition(v_uxp, v_uxm);
            basis_value* uc_2 = basis_value_multiplication(v_uc, basis_value_new(2.0));
            basis_value* dxx_2 = basis_value_subtraction(dxx_1, uc_2);
            basis_value* dxx_scale = basis_value_new(1.0 / (EPS * EPS));
            basis_value* dxx_final = basis_value_multiplication(dxx_2, dxx_scale);

            basis_value* res = basis_value_subtraction(dt_final, dxx_final);
            basis_value* res_sq = basis_value_power(res, 2.0);

            basis_value* new_sum = basis_value_addition(loss_sum, res_sq);
            basis_value_free(loss_sum); loss_sum = new_sum;

            basis_value_free(v_uc); basis_value_free(v_utp); basis_value_free(v_utm);
            basis_value_free(v_uxp); basis_value_free(v_uxm);
            basis_value_free(dt_u); basis_value_free(dt_scale); basis_value_free(dt_final);
            basis_value_free(dxx_1); basis_value_free(uc_2); basis_value_free(dxx_2);
            basis_value_free(dxx_scale); basis_value_free(dxx_final);
            basis_value_free(res); basis_value_free(res_sq);
        }

        // 2. Initial Condition Loss
        for(int i=0; i<NUM_IC; i++) {
            double xc = x_i[i];
            basis_tensor* u_ic = eval_u(xc, 0.0, W1, b1, W2, b2);
            basis_value* v_uic = u_ic->data[0]; basis_value_retain(v_uic);
            basis_tensor_free(u_ic);

            basis_value* target = basis_value_new(sin(PI * xc));
            basis_value* diff = basis_value_subtraction(v_uic, target);
            basis_value* sq = basis_value_power(diff, 2.0);

            basis_value* new_sum = basis_value_addition(loss_sum, sq);
            basis_value_free(loss_sum); loss_sum = new_sum;

            basis_value_free(v_uic); basis_value_free(target); basis_value_free(diff); basis_value_free(sq);
        }

        basis_value* divisor = basis_value_new(1.0 / (NUM_COLLOC + NUM_IC));
        basis_value* loss = basis_value_multiplication(loss_sum, divisor);

        if (epoch % 100 == 0) printf("Epoch %4d | Loss: %.6f\n", epoch, loss->data);

        basis_value_backward_propagation(loss);

        basis_adam_optimization_step(opt_W1, W1);
        basis_adam_optimization_step(opt_b1, b1);
        basis_adam_optimization_step(opt_W2, W2);
        basis_adam_optimization_step(opt_b2, b2);

        basis_value_free(loss); basis_value_free(divisor); basis_value_free(loss_sum);
    }

    printf("\n=== Final Verification ===\n");
    printf("x\t | t=0.0 (Target: sin(pi*x)) \t | t=0.1 (Predicted)\n");
    for(int i=0; i<=10; i++) {
        double xc = i / 10.0;
        basis_tensor* u_0 = eval_u(xc, 0.0, W1, b1, W2, b2);
        basis_tensor* u_1 = eval_u(xc, 0.1, W1, b1, W2, b2);
        printf("%.1f\t | %.4f \t\t\t\t | %.4f\n", xc, u_0->data[0]->data, u_1->data[0]->data);
        basis_tensor_free(u_0); basis_tensor_free(u_1);
    }

    basis_adam_free(opt_W1); basis_adam_free(opt_b1); basis_adam_free(opt_W2); basis_adam_free(opt_b2);
    basis_tensor_free(W1); basis_tensor_free(b1); basis_tensor_free(W2); basis_tensor_free(b2);
    return 0;
}
