/**
 * @file lancius/lancius_alu.h
 * @brief LAIS — Lancius ALU Instruction Set.
 *
 * Generation 1: control plane (Unix domain socket, text commands) +
 * data plane (POSIX shared memory, float64 tensors).
 *
 * The CU (Control Unit) is an external language model. The ALUs are
 * Lancius static-graph verifiers. The CU speaks natural language;
 * the ALUs speak LAIS.
 */

/* Socket path template: /tmp/lancius_alu_<id>.sock */
/* SHM name template:    /lancius_alu_<id>          */

#define LANCIUS_ALU_SOCK_PATH   "/tmp/lancius_alu_%u.sock"
#define LANCIUS_ALU_SHM_NAME    "/lancius_alu_%u"

#define LANCIUS_ALU_SHM_MAGIC   0x414c5553u   /* "ALUS" */
#define LANCIUS_ALU_MAX_ALUS    32
#define LANCIUS_ALU_MAX_MODEL   4096
#define LANCIUS_ALU_MAX_LINE    2048

/* -------- Opcodes (sent by CU to ALU) -------- */
typedef enum {
    LAIS_OP_NOP        = 0x00,
    LAIS_OP_LOAD_MODEL = 0x01,   /* load .lancius into ALU slot            */
    LAIS_OP_VERIFY     = 0x02,   /* run verification on shm-resident data  */
    LAIS_OP_STATUS     = 0x03,   /* ALU health / model identity            */
    LAIS_OP_REDUCE     = 0x04,   /* combine N ALU verdicts (AND/OR/MAJ)    */
    LAIS_OP_QUIT       = 0x05,   /* graceful shutdown                      */
    LAIS_OP_ECHO       = 0x06    /* round-trip test                        */
} lais_opcode;

/* -------- Reduce operations -------- */
typedef enum {
    LAIS_REDUCE_AND    = 0x00,   /* all +1 => +1, any -1 => -1             */
    LAIS_REDUCE_OR     = 0x01,   /* any +1 => +1, all -1 => -1             */
    LAIS_REDUCE_MAJ    = 0x02    /* majority vote                          */
} lais_reduce_op;

/* -------- Verdict mapping (Gen1 5-class) -------- */
/* Raw output class -> LAIS int8 verdict -> English meaning */
/*   -1.0  -> -100  "False — derivation broken"      */
/*   -0.5  ->  -50  "Mostly false, something right"  */
/*    0.0  ->    0  "Indeterminate / neutral"        */
/*   +0.5  ->  +50  "Mostly true, something wrong"   */
/*   +1.0  -> +100  "True — derivation sound"        */
#define LANCIUS_ALU_VERDICT_FALSE   (-100)
#define LANCIUS_ALU_VERDICT_MOSTLY_FALSE (-50)
#define LANCIUS_ALU_VERDICT_NEUTRAL       (  0)
#define LANCIUS_ALU_VERDICT_MOSTLY_TRUE   ( 50)
#define LANCIUS_ALU_VERDICT_TRUE          (100)

/* -------- Shared memory header (one per ALU) -------- */
typedef struct {
    uint32_t magic;            /* LANCIUS_ALU_SHM_MAGIC                      */
    uint32_t status;           /* 0=empty, 1=ready, 2=done, 3=error          */
    uint32_t input_elems;      /* float64 count                              */
    uint32_t input_bytes;      /* input_elems * 8                            */
    int32_t  verdict;          /* ALU response verdict (int8 range)          */
    float    confidence;       /* response confidence (max |logit| margin)   */
    char     model_id[64];     /* architecture string, e.g. "65->4096->5"   */
    char     pad[4];           /* alignment                                  */
} __attribute__((packed)) lancius_alu_shm_header;

#define LANCIUS_ALU_SHM_HEADER_SIZE \
    (sizeof(lancius_alu_shm_header) + 8)  /* 8-byte aligned */

/* -------- Socket command / response (text, newline-terminated) -------- */
/* CU -> ALU:  "LOAD 3 /path/to/model.lancius\n" */
/* CU -> ALU:  "VERIFY 3 /lancius_alu_3 200\n" */
/* CU -> ALU:  "STATUS 3\n" */
/* CU -> ALU:  "REDUCE MAJ 3 1 2 3\n" */
/* CU -> ALU:  "QUIT 3\n" */
/* ALU -> CU:  "OK 100 0.940 65->4096->2560->5\n" */
/* ALU -> CU:  "ERR -9 NUMERICAL: softmax zero-sum\n" */

#endif /* LANCIUS_ALU_H */