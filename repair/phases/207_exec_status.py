#!/usr/bin/env python3
"""Phase 2.7: Make execute_node_math return a status and propagate errors.
Kills the silent-return-on-failure pattern that causes train-on-garbage."""
import os, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_harness"))
import safe_edit as se

FILE = "src/runtime/lancius_scheduler.c"
MARKER = "v12R1-207"

def main():
    print("[207] execution status propagation")
    content = se.read(se.p(FILE))
    if MARKER in content:
        print(f"  [skip] {FILE}: already applied")
        return

    # 1. Change execute_node_math signature from void to int
    content = se.replace_once(
        content,
        "static void execute_node_math(lancius_node* n) {",
        f"/* {MARKER}: returns 0 on success, non-zero on failure */\n"
        "static int execute_node_math(lancius_node* n) {"
    )

    # 2. Change the initial null check to return error
    content = se.replace_once(
        content,
        "static int execute_node_math(lancius_node* n) {\n    if (!n) return;",
        "static int execute_node_math(lancius_node* n) {\n    if (!n) return -1;"
    )

    # 3. Change FP32 unsupported dtype return to return error
    content = se.replace_once(
        content,
        "            lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_DTYPE);\n            return;",
        "            lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_DTYPE);\n            return -1;"
    )

    # 4. Change the null runtime_data return to return error
    content = se.replace_once(
        content,
        "    } else if (!n->runtime_data) {\n        return;\n    }",
        "    } else if (!n->runtime_data) {\n        return -1;\n    }"
    )

    # 5. Change all "if (!a || !b) return;" to return -1
    content = content.replace("if (!a || !b) return;", "if (!a || !b) return -1;")

    # 6. Change all "if (!a) return;" in execute_node_math to return -1
    # (only within execute_node_math, not other functions)
    # We'll do this by finding the function body and replacing within it
    func_start = content.find("static int execute_node_math(lancius_node* n) {")
    if func_start == -1:
        raise RuntimeError("[ABORT] could not find execute_node_math")

    # Find the end of the function (next static function or end of file)
    func_end = content.find("\nstatic void lancius_schedule_prepare_buffers", func_start)
    if func_end == -1:
        func_end = content.find("\nstatic void execute_permute", func_start)
    if func_end == -1:
        func_end = len(content)

    func_body = content[func_start:func_end]

    # Replace silent returns within the function body
    func_body = func_body.replace("if (!a) return;", "if (!a) return -1;")
    func_body = func_body.replace("if (!in) return;", "if (!in) return -1;")
    func_body = func_body.replace("if (!x || !y) return;", "if (!x || !y) return -1;")
    func_body = func_body.replace("if(!in || !gamma || !beta) return;", "if(!in || !gamma || !beta) return -1;")
    func_body = func_body.replace("if(!in || !gamma) return;", "if(!in || !gamma) return -1;")
    func_body = func_body.replace("if(!gate || !up) return;", "if(!gate || !up) return -1;")
    func_body = func_body.replace("if(!q || !k || !v) return;", "if(!q || !k || !v) return -1;")
    func_body = func_body.replace("if (!grad || !fwd_a) return;", "if (!grad || !fwd_a) return -1;")
    func_body = func_body.replace("if (!dy || !y) return;", "if (!dy || !y) return -1;")
    func_body = func_body.replace("if (!grad || !w) return;", "if (!grad || !w) return -1;")
    func_body = func_body.replace("if (!grad || !in) return;", "if (!grad || !in) return -1;")
    func_body = func_body.replace("if (!grad || !fwd_in) return;", "if (!grad || !fwd_in) return -1;")
    func_body = func_body.replace("if(!in || !w) return;", "if(!in || !w) return -1;")

    # Add "return 0;" at the end of the function (before the closing brace)
    # Find the last closing brace of the function
    func_body = func_body.rstrip()
    if func_body.endswith("}"):
        func_body = func_body[:-1] + "\n    return 0;\n}"

    content = content[:func_start] + func_body + content[func_end:]

    # 7. Change the cross-entropy "return;" statements to "return 0;"
    # These are within execute_node_math and should return success
    content = content.replace(
        "        if (n->runtime_data[0] < 0.0 && n->runtime_data[0] > -1e-12) n->runtime_data[0] = 0.0;\n        return;",
        "        if (n->runtime_data[0] < 0.0 && n->runtime_data[0] > -1e-12) n->runtime_data[0] = 0.0;\n        return 0;"
    )

    # 8. Change the cross-entropy-bwd return
    content = content.replace(
        "            n->runtime_data[r*C+c] = (sm - y[r*C+c]) * scale;\n        }\n    }\n    return;",
        "            n->runtime_data[r*C+c] = (sm - y[r*C+c]) * scale;\n        }\n    }\n    return 0;"
    )

    # 9. Change execute_permute and execute_matmul_batched to return int
    content = se.replace_once(
        content,
        "static void execute_permute(lancius_node* n) {",
        "static int execute_permute(lancius_node* n) {"
    )
    content = content.replace(
        "static int execute_permute(lancius_node* n) {\n    if (!n || n->input_count == 0) return;",
        "static int execute_permute(lancius_node* n) {\n    if (!n || n->input_count == 0) return -1;"
    )
    content = content.replace(
        "    if (!x || !y) return;\n\n    size_t in_shape[4]",
        "    if (!x || !y) return -1;\n\n    size_t in_shape[4]"
    )
    # Add return 0 at end of execute_permute
    # Find the end of execute_permute (before execute_matmul_batched)
    perm_end = content.find("static int execute_matmul_batched")
    if perm_end != -1:
        # Find the closing brace before it
        search_region = content[content.find("static int execute_permute"):perm_end]
        # The function ends with the closing brace of the outer for loop
        # We need to add return 0 before the final }
        # Find the last } before execute_matmul_batched
        last_brace = content.rfind("}", content.find("static int execute_permute"), perm_end)
        if last_brace != -1:
            content = content[:last_brace] + "    return 0;\n}" + content[last_brace+1:]

    content = se.replace_once(
        content,
        "static void execute_matmul_batched(lancius_node* n) {",
        "static int execute_matmul_batched(lancius_node* n) {"
    )
    content = content.replace(
        "static int execute_matmul_batched(lancius_node* n) {\n    if (!n || n->input_count < 2) return;",
        "static int execute_matmul_batched(lancius_node* n) {\n    if (!n || n->input_count < 2) return -1;"
    )
    content = content.replace(
        "    if (!A || !B || !C) return;\n\n    size_t batches",
        "    if (!A || !B || !C) return -1;\n\n    size_t batches"
    )
    # Add return 0 at end of execute_matmul_batched
    mmb_end = content.find("size_t lancius_schedule_static_memory_required")
    if mmb_end != -1:
        last_brace = content.rfind("}", content.find("static int execute_matmul_batched"), mmb_end)
        if last_brace != -1:
            content = content[:last_brace] + "    return 0;\n}" + content[last_brace+1:]

    # 10. Make the scheduler check execute_node_math return value
    # In lancius_schedule_execute, change the call to check return
    content = content.replace(
        "            execute_node_math(n);\n        }\n    }\n}",
        f"            /* {MARKER}: stop on first failure */\n"
        "            if (execute_node_math(n) != 0) {\n"
        "                lancius_set_error(LANCIUS_ERROR_INTERNAL);\n"
        "                return;\n"
        "            }\n"
        "        }\n    }\n}"
    )

    # 11. Same for the static executor
    content = content.replace(
        "            execute_node_math(n);\n        }\n    }\n}\n\n/* v11S H2 fix",
        f"            /* {MARKER}: stop on first failure */\n"
        "            if (execute_node_math(n) != 0) {\n"
        "                lancius_set_error(LANCIUS_ERROR_INTERNAL);\n"
        "                return;\n"
        "            }\n"
        "        }\n    }\n}\n\n/* v11S H2 fix"
    )

    se.backup(se.p(FILE))
    se.write(se.p(FILE), content)
    print(f"  [edit] {FILE}")
    print("[207] done")

if __name__ == "__main__":
    main()
