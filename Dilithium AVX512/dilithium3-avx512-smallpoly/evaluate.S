//preprocessor macro
#if defined(__WIN32__) || defined(__APPLE__)
#define cdecl(s) _##s
#else
#define cdecl(s) s
#endif


.macro accanswer off
vmovdqu32   (\off)(%rsi),%zmm0
vmovdqu32   (64+\off)(%rsi),%zmm1
vmovdqu32   (128+\off)(%rsi),%zmm2
vmovdqu32   (192+\off)(%rsi),%zmm3
vmovdqu32   (256+\off)(%rsi),%zmm4
vmovdqu32   (320+\off)(%rsi),%zmm5
vmovdqu32   (384+\off)(%rsi),%zmm6
vmovdqu32   (448+\off)(%rsi),%zmm7
vmovdqu32   (\off)(%rdi),%zmm8
vmovdqu32   (64+\off)(%rdi),%zmm9
vmovdqu32   (128+\off)(%rdi),%zmm10
vmovdqu32   (192+\off)(%rdi),%zmm11
vmovdqu32   (256+\off)(%rdi),%zmm12
vmovdqu32   (320+\off)(%rdi),%zmm13
vmovdqu32   (384+\off)(%rdi),%zmm14
vmovdqu32   (448+\off)(%rdi),%zmm15

vpaddd    %zmm0,%zmm8,%zmm0
vpaddd    %zmm1,%zmm9,%zmm1
vpaddd    %zmm2,%zmm10,%zmm2
vpaddd    %zmm3,%zmm11,%zmm3

vpaddd    %zmm4,%zmm12,%zmm4
vpaddd    %zmm5,%zmm13,%zmm5
vpaddd    %zmm6,%zmm14,%zmm6
vpaddd    %zmm7,%zmm15,%zmm7

vmovdqu32   %zmm0,(\off)(%rdi)
vmovdqu32   %zmm1,(64+\off)(%rdi)
vmovdqu32   %zmm2,(128+\off)(%rdi)
vmovdqu32   %zmm3,(192+\off)(%rdi)
vmovdqu32   %zmm4,(256+\off)(%rdi)
vmovdqu32   %zmm5,(320+\off)(%rdi)
vmovdqu32   %zmm6,(384+\off)(%rdi)
vmovdqu32   %zmm7,(448+\off)(%rdi)

.endm

.p2align 5   //instructs the assembler to align the following instruction or data on a boundary that is a power of 2 and here is equal to 2^5, or 32 bytes
.global cdecl(add_asm)
cdecl(add_asm):

accanswer       0
accanswer       512
ret




.macro accmaskanswer off
vmovdqu32   (\off)(%rsi),%zmm0
vmovdqu32   (64+\off)(%rsi),%zmm1
vmovdqu32   (128+\off)(%rsi),%zmm2
vmovdqu32   (192+\off)(%rsi),%zmm3
vmovdqu32   (256+\off)(%rsi),%zmm4
vmovdqu32   (320+\off)(%rsi),%zmm5
vmovdqu32   (384+\off)(%rsi),%zmm6
vmovdqu32   (448+\off)(%rsi),%zmm7
vmovdqu32   (\off)(%rdi),%zmm8
vmovdqu32   (64+\off)(%rdi),%zmm9
vmovdqu32   (128+\off)(%rdi),%zmm10
vmovdqu32   (192+\off)(%rdi),%zmm11
vmovdqu32   (256+\off)(%rdi),%zmm12
vmovdqu32   (320+\off)(%rdi),%zmm13
vmovdqu32   (384+\off)(%rdi),%zmm14
vmovdqu32   (448+\off)(%rdi),%zmm15

vpsubd    %zmm0,%zmm30,%zmm0
vpsubd    %zmm1,%zmm30,%zmm1
vpsubd    %zmm2,%zmm30,%zmm2
vpsubd    %zmm3,%zmm30,%zmm3
vpsubd    %zmm4,%zmm30,%zmm4
vpsubd    %zmm5,%zmm30,%zmm5
vpsubd    %zmm6,%zmm30,%zmm6
vpsubd    %zmm7,%zmm30,%zmm7

vpaddd    %zmm0,%zmm8,%zmm0
vpaddd    %zmm1,%zmm9,%zmm1
vpaddd    %zmm2,%zmm10,%zmm2
vpaddd    %zmm3,%zmm11,%zmm3
vpaddd    %zmm4,%zmm12,%zmm4
vpaddd    %zmm5,%zmm13,%zmm5
vpaddd    %zmm6,%zmm14,%zmm6
vpaddd    %zmm7,%zmm15,%zmm7


vmovdqu32   %zmm0,(\off)(%rdi)
vmovdqu32   %zmm1,(64+\off)(%rdi)
vmovdqu32   %zmm2,(128+\off)(%rdi)
vmovdqu32   %zmm3,(192+\off)(%rdi)
vmovdqu32   %zmm4,(256+\off)(%rdi)
vmovdqu32   %zmm5,(320+\off)(%rdi)
vmovdqu32   %zmm6,(384+\off)(%rdi)
vmovdqu32   %zmm7,(448+\off)(%rdi)

.endm

.p2align 5   //instructs the assembler to align the following instruction or data on a boundary that is a power of 2 and here is equal to 2^5, or 32 bytes
.global cdecl(addmask11_asm)
cdecl(addmask11_asm):

vpbroadcastd		_16xmasks11(%rip),%zmm30
accmaskanswer       0
accmaskanswer       512

ret

.p2align 5   //instructs the assembler to align the following instruction or data on a boundary that is a power of 2 and here is equal to 2^5, or 32 bytes
.global cdecl(addmask12_asm)
cdecl(addmask12_asm):

vpbroadcastd		_16xmasks12(%rip),%zmm30
accmaskanswer       0
accmaskanswer       512

ret

