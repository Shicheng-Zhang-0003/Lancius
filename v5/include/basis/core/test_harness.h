
#ifndef BASIS_TEST_HARNESS_H
#define BASIS_TEST_HARNESS_H
#include <stdio.h>
#include <math.h>

static int basis_tests_run = 0;
static int basis_tests_failed = 0;

#define BASIS_ASSERT(cond, msg) do { \
    basis_tests_run++; \
    if (!(cond)) { \
        printf("  [FAIL] %s (Line %d): %s\n", __func__, __LINE__, msg); \
        basis_tests_failed++; \
    } else { \
        printf("  [PASS] %s\n", msg); \
    } \
} while(0)

#define BASIS_ASSERT_NEAR(a, b, eps, msg) do { \
    basis_tests_run++; \
    if (fabs((a) - (b)) > (eps)) { \
        printf("  [FAIL] %s (Line %d): %s (Expected %f, Got %f)\n", __func__, __LINE__, msg, (double)(b), (double)(a)); \
        basis_tests_failed++; \
    } else { \
        printf("  [PASS] %s\n", msg); \
    } \
} while(0)

#define BASIS_TEST_SUITE_START(name) printf("\n=== SUITE: %s ===\n", name);
#define BASIS_TEST_SUITE_END() do { \
    printf("\nResults: %d/%d passed.\n", basis_tests_run - basis_tests_failed, basis_tests_run); \
    if (basis_tests_failed > 0) printf("!!! FAILURES DETECTED !!!\n"); \
    else printf("✅ ALL TESTS PASSED.\n"); \
} while(0)
#endif
