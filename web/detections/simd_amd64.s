#include "textflag.h"

// Constants
DATA float255inv<>+0(SB)/4, $0x3B808081  // 1.0/255.0
GLOBL float255inv<>(SB), RODATA, $4

// AVX-512 implementation
TEXT ·processRowAVX512(SB), NOSPLIT, $0-24
    MOVQ    dst+0(FP), DI
    MOVQ    src+8(FP), SI
    MOVQ    width+16(FP), CX

    VBROADCASTSS float255inv<>+0(SB), Z15

    // Process 16 pixels at a time
    SHRQ    $4, CX                     // Divide width by 16
    JZ      avx512_remainder

avx512_loop:
    // Load and process pixels
    VPMOVZXBD   (SI), Z0              // Zero extend bytes to dwords
    VPBROADCASTD float255inv<>+0(SB), Z1
    VPMULLD     Z0, Z1, Z0
    VMOVUPS     Z0, (DI)              // Store result

    ADDQ    $16, SI                    // Advance source pointer
    ADDQ    $64, DI                    // Advance destination pointer
    DECQ    CX
    JNZ     avx512_loop

avx512_remainder:
    MOVQ    width+16(FP), CX
    ANDQ    $15, CX                    // Get remainder
    JZ      avx512_done

avx512_remainder_loop:
    MOVBQZX (SI), AX                   // Load byte
    MOVL    AX, X0
    MULSS   float255inv<>+0(SB), X0
    MOVSS   X0, (DI)

    INCQ    SI
    ADDQ    $4, DI
    DECQ    CX
    JNZ     avx512_remainder_loop

avx512_done:
    VZEROUPPER
    RET

// AVX2 implementation
TEXT ·processRowAVX2(SB), NOSPLIT, $0-24
    MOVQ    dst+0(FP), DI
    MOVQ    src+8(FP), SI
    MOVQ    width+16(FP), CX

    VBROADCASTSS float255inv<>+0(SB), Y15

    // Process 8 pixels at a time
    SHRQ    $3, CX                     // Divide width by 8
    JZ      avx2_remainder

avx2_loop:
    // Load and process pixels
    VPMOVZXBD   (SI), Y0              // Zero extend bytes to dwords
    VPBROADCASTD float255inv<>+0(SB), Y1
    VPMULLD     Y0, Y1, Y0
    VMOVUPS     Y0, (DI)              // Store result

    ADDQ    $8, SI
    ADDQ    $32, DI
    DECQ    CX
    JNZ     avx2_loop

avx2_remainder:
    MOVQ    width+16(FP), CX
    ANDQ    $7, CX                     // Get remainder
    JZ      avx2_done

avx2_remainder_loop:
    MOVBQZX (SI), AX                   // Load byte
    MOVL    AX, X0
    MULSS   float255inv<>+0(SB), X0
    MOVSS   X0, (DI)

    INCQ    SI
    ADDQ    $4, DI
    DECQ    CX
    JNZ     avx2_remainder_loop

avx2_done:
    VZEROUPPER
    RET

// SSE4.1 implementation
TEXT ·processRowSSE41(SB), NOSPLIT, $0-24
    MOVQ    dst+0(FP), DI
    MOVQ    src+8(FP), SI
    MOVQ    width+16(FP), CX

    MOVSS   float255inv<>+0(SB), X15
    SHUFPS  $0, X15, X15               // Broadcast to all elements

    // Process 4 pixels at a time
    SHRQ    $2, CX                     // Divide width by 4
    JZ      sse_remainder

sse_loop:
    // Load and process pixels
    PMOVZXBD    (SI), X0              // Zero extend bytes to dwords
    MOVSS       float255inv<>+0(SB), X1
    SHUFPS      $0, X1, X1
    MULPS       X1, X0
    MOVUPS      X0, (DI)              // Store result

    ADDQ    $4, SI
    ADDQ    $16, DI
    DECQ    CX
    JNZ     sse_loop

sse_remainder:
    MOVQ    width+16(FP), CX
    ANDQ    $3, CX                     // Get remainder
    JZ      sse_done

sse_remainder_loop:
    MOVBQZX (SI), AX                   // Load byte
    MOVL    AX, X0
    MULSS   float255inv<>+0(SB), X0
    MOVSS   X0, (DI)

    INCQ    SI
    ADDQ    $4, DI
    DECQ    CX
    JNZ     sse_remainder_loop

sse_done:
    RET
