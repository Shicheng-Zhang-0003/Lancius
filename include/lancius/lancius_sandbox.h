#ifndef LANCIUS_SANDBOX_H
#define LANCIUS_SANDBOX_H

/* Minimal honest sandbox: static resource gate + weight finiteness gate.
 *
 * Contract:
 *  - lancius_sandbox_check_graph: 0 = ACCEPT, -1 = REJECT (do not
 *    schedule/execute). Sets lancius error on reject, clears it on entry.
 *  - lancius_sandbox_check_weights: 0 = ok, -1 = ABSTAIN (untrusted model:
 *    non-finite or unmeasurable weight buffer; do not execute, do not
 *    crash). Sets lancius error on -1, clears it on entry.
 */

#include "lancius/lancius_ir.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t max_bytes;
    uint32_t max_nodes;
    uint32_t max_steps;
} lancius_sandbox_caps;

int lancius_sandbox_check_graph(const lancius_graph *g,
                                const lancius_sandbox_caps *caps);

int lancius_sandbox_check_weights(const lancius_graph *g);

#ifdef __cplusplus
}
#endif

#endif /* LANCIUS_SANDBOX_H */
