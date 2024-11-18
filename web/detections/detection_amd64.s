#include "textflag.h"

// Constants
DATA float255inv<>+0(SB)/4, $0x3B808081  // 1.0/255.0
GLOBL float255inv<>(SB), RODATA, $4

// func processRowAVX2(dst, src unsafe.Pointer, width int)
TEXT ·processRowAVX2(SB), NOSPLIT, $0-24
    MOVQ    dst+0(FP), DI    // Load destination pointer
    MOVQ    src+8(FP), SI    // Load source pointer
    MOVQ    width+16(FP), CX // Load width

    VBROADCASTSS float255inv<>+0(SB), Y15 // Load 1/255.0 into all lanes

loop:
    CMPQ    CX, $8
    JL      tail

    // Load 8 pixels
    VPMOVZXBD  (SI), Y0   // Zero extend 8 bytes to 8 int32
    VCVTDQ2PS  Y0, Y0     // Convert int32 to float32
    VMULPS     Y15, Y0, Y0 // Multiply by 1/255.0

    // Store result
    VMOVUPS Y0, (DI)

    // Advance pointers and counter
    ADDQ    $8, SI
    ADDQ    $32, DI  // 8 float32s = 32 bytes
    SUBQ    $8, CX
    JNZ     loop

tail:
    TESTQ   CX, CX
    JZ      done

tail_loop:
    // Handle remaining pixels one at a time
    MOVBQZX  (SI), AX    // Changed from MOVZXB to MOVBQZX
    VCVTSI2SSL AX, X0, X0 // Changed from VCVTSI2SS to VCVTSI2SSL
    VMULSS  float255inv<>+0(SB), X0, X0
    VMOVSS  X0, (DI)

    INCQ    SI
    ADDQ    $4, DI
    DECQ    CX
    JNZ     tail_loop

done:
    VZEROUPPER
    RET
