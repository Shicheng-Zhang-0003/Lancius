#include <basis.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand(42);
    printf("=== BASIS v5: End-to-End Deployment Pipeline ===\n\n");

    // ==========================================
    // PHASE 1: TRAINING (The "Lab" Environment)
    // ==========================================
    printf("[1/4] Training XOR Network...\n");
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

    for(size_t i=0; i<16; i++) W1->data[i]->data = ((rand() % 100) / 100.0) - 0.5;
    for(size_t i=0; i<8; i++)  W2->data[i]->data = ((rand() % 100) / 100.0) - 0.5;

    basis_adam* opt_W1 = basis_adam_new(2, 8, 0.05);
    basis_adam* opt_b1 = basis_adam_new(1, 8, 0.05);
    basis_adam* opt_W2 = basis_adam_new(8, 1, 0.05);
    basis_adam* opt_b2 = basis_adam_new(1, 1, 0.05);

    for(int epoch = 0; epoch < 2000; epoch++) {
        basis_tensor* Z1 = basis_tensor_matrix_multiplication(X, W1);
        basis_tensor* b1_view = basis_tensor_broadcast_view(b1, 4, 8);
        basis_tensor* A1 = basis_tensor_add(Z1, b1_view);
        basis_tensor* H1 = basis_tensor_rectified_linear_unit(A1);
        basis_tensor* Z2 = basis_tensor_matrix_multiplication(H1, W2);
        basis_tensor* b2_view = basis_tensor_broadcast_view(b2, 4, 1);
        basis_tensor* Y_pred = basis_tensor_add(Z2, b2_view);

        basis_value* loss_sum = basis_value_new(0.0);
        for(int i=0; i<4; i++) {
            basis_value* diff = basis_value_subtraction(Y_pred->data[i], Y_true->data[i]);
            basis_value* sq = basis_value_power(diff, 2.0);
            basis_value* new_sum = basis_value_addition(loss_sum, sq);
            basis_value_free(loss_sum); basis_value_free(diff); basis_value_free(sq);
            loss_sum = new_sum;
        }
        basis_value* divisor = basis_value_new(0.25);
        basis_value* loss = basis_value_multiplication(loss_sum, divisor);

        basis_value_backward_propagation(loss);

        basis_adam_optimization_step(opt_W1, W1); basis_adam_optimization_step(opt_b1, b1);
        basis_adam_optimization_step(opt_W2, W2); basis_adam_optimization_step(opt_b2, b2);

        basis_value_free(loss); basis_value_free(divisor); basis_value_free(loss_sum);
        basis_tensor_free(Y_pred); basis_tensor_free(b2_view); basis_tensor_free(Z2);
        basis_tensor_free(H1); basis_tensor_free(A1); basis_tensor_free(b1_view); basis_tensor_free(Z1);
    }
    printf("      Training complete.\n");

    // ==========================================
    // PHASE 2: SERIALIZATION (Export to Disk)
    // ==========================================
    printf("[2/4] Saving weights to disk (*.basis)...\n");
    basis_tensor_save_binary(W1, "W1.basis");
    basis_tensor_save_binary(b1, "b1.basis");
    basis_tensor_save_binary(W2, "W2.basis");
    basis_tensor_save_binary(b2, "b2.basis");
    printf("      Weights exported to binary format.\n");

    // ==========================================
    // PHASE 3: TEARDOWN (Simulate shutting down the training server)
    // ==========================================
    printf("[3/4] Purging training memory...\n");
    basis_adam_free(opt_W1); basis_adam_free(opt_b1);
    basis_adam_free(opt_W2); basis_adam_free(opt_b2);
    basis_tensor_free(W1); basis_tensor_free(b1);
    basis_tensor_free(W2); basis_tensor_free(b2);
    basis_tensor_free(X); basis_tensor_free(Y_true);
    printf("      All training tensors freed. Memory is empty.\n\n");

    // ==========================================
    // PHASE 4: DEPLOYMENT (Load & Infer)
    // ==========================================
    printf("[4/4] Loading weights into fresh memory for Inference...\n");
    basis_tensor* W1_inf = basis_tensor_load_binary("W1.basis");
    basis_tensor* b1_inf = basis_tensor_load_binary("b1.basis");
    basis_tensor* W2_inf = basis_tensor_load_binary("W2.basis");
    basis_tensor* b2_inf = basis_tensor_load_binary("b2.basis");
    printf("      Weights successfully loaded from disk.\n\n");

    printf("=== Inference Results ===\n");
    basis_tensor* X_test = basis_tensor_new(4, 2);
    basis_tensor_set(X_test, 0, 0, 0.0); basis_tensor_set(X_test, 0, 1, 0.0);
    basis_tensor_set(X_test, 1, 0, 0.0); basis_tensor_set(X_test, 1, 1, 1.0);
    basis_tensor_set(X_test, 2, 0, 1.0); basis_tensor_set(X_test, 2, 1, 0.0);
    basis_tensor_set(X_test, 3, 0, 1.0); basis_tensor_set(X_test, 3, 1, 1.0);

    basis_tensor* Z1 = basis_tensor_matrix_multiplication(X_test, W1_inf);
    basis_tensor* b1_view = basis_tensor_broadcast_view(b1_inf, 4, 8);
    basis_tensor* A1 = basis_tensor_add(Z1, b1_view);
    basis_tensor* H1 = basis_tensor_rectified_linear_unit(A1);
    basis_tensor* Z2 = basis_tensor_matrix_multiplication(H1, W2_inf);
    basis_tensor* b2_view = basis_tensor_broadcast_view(b2_inf, 4, 1);
    basis_tensor* Y_pred = basis_tensor_add(Z2, b2_view);

    for(int i=0; i<4; i++) {
        printf("Input: [%.0f, %.0f] | Predicted: %.4f\n",
               X_test->data[i*2]->data, X_test->data[i*2+1]->data,
               Y_pred->data[i]->data);
    }

    basis_tensor_free(Y_pred); basis_tensor_free(b2_view); basis_tensor_free(Z2);
    basis_tensor_free(H1); basis_tensor_free(A1); basis_tensor_free(b1_view); basis_tensor_free(Z1);
    basis_tensor_free(W1_inf); basis_tensor_free(b1_inf);
    basis_tensor_free(W2_inf); basis_tensor_free(b2_inf);
    basis_tensor_free(X_test);

    printf("\n✅ Deployment pipeline complete. Zero memory leaks.\n");
    return 0;
}
