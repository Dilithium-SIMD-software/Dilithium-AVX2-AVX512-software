//preprocessor macro
#if defined(__WIN32__) || defined(__APPLE__)
#define cdecl(s) _##s
#else
#define cdecl(s) s
#endif



.p2align 5   //instructs the assembler to align the following instruction or data on a boundary that is a power of 2 and here is equal to 2^5, or 32 bytes
.global cdecl(add_asm)
cdecl(add_asm):

//load answer[0]-answer[255]
vmovdqu32   (%rdi),%zmm0
vmovdqu32   (64)(%rdi),%zmm1
vmovdqu32   (128)(%rdi),%zmm2
vmovdqu32   (192)(%rdi),%zmm3
vmovdqu32   (256)(%rdi),%zmm4
vmovdqu32   (320)(%rdi),%zmm5
vmovdqu32   (384)(%rdi),%zmm6
vmovdqu32   (448)(%rdi),%zmm7
vmovdqu32   (512)(%rdi),%zmm8
vmovdqu32   (576)(%rdi),%zmm9
vmovdqu32   (640)(%rdi),%zmm10
vmovdqu32   (704)(%rdi),%zmm11
vmovdqu32   (768)(%rdi),%zmm12
vmovdqu32   (832)(%rdi),%zmm13
vmovdqu32   (896)(%rdi),%zmm14
vmovdqu32   (960)(%rdi),%zmm15
//load s_table[N-i]-s_table[255+N+i]
vmovdqu32   (%rsi),%zmm16
vmovdqu32   (64)(%rsi),%zmm17
vmovdqu32   (128)(%rsi),%zmm18
vmovdqu32   (192)(%rsi),%zmm19
vmovdqu32   (256)(%rsi),%zmm20
vmovdqu32   (320)(%rsi),%zmm21
vmovdqu32   (384)(%rsi),%zmm22
vmovdqu32   (448)(%rsi),%zmm23
vmovdqu32   (512)(%rsi),%zmm24
vmovdqu32   (576)(%rsi),%zmm25
vmovdqu32   (640)(%rsi),%zmm26
vmovdqu32   (704)(%rsi),%zmm27
vmovdqu32   (768)(%rsi),%zmm28
vmovdqu32   (832)(%rsi),%zmm29
vmovdqu32   (896)(%rsi),%zmm30
vmovdqu32   (960)(%rsi),%zmm31
//add
vpaddd       %zmm0,%zmm16,%zmm0
vpaddd       %zmm1,%zmm17,%zmm1
vpaddd       %zmm2,%zmm18,%zmm2
vpaddd       %zmm3,%zmm19,%zmm3
vpaddd       %zmm4,%zmm20,%zmm4
vpaddd       %zmm5,%zmm21,%zmm5
vpaddd       %zmm6,%zmm22,%zmm6
vpaddd       %zmm7,%zmm23,%zmm7
vpaddd       %zmm8,%zmm24,%zmm8
vpaddd       %zmm9,%zmm25,%zmm9
vpaddd       %zmm10,%zmm26,%zmm10
vpaddd       %zmm11,%zmm27,%zmm11
vpaddd       %zmm12,%zmm28,%zmm12
vpaddd       %zmm13,%zmm29,%zmm13
vpaddd       %zmm14,%zmm30,%zmm14
vpaddd       %zmm15,%zmm31,%zmm15
//store answer[0]-answer[255]
vmovdqu32   %zmm0,(0)(%rdi)
vmovdqu32   %zmm1,(64)(%rdi)
vmovdqu32   %zmm2,(128)(%rdi)
vmovdqu32   %zmm3,(192)(%rdi)
vmovdqu32   %zmm4,(256)(%rdi)
vmovdqu32   %zmm5,(320)(%rdi)
vmovdqu32   %zmm6,(384)(%rdi)
vmovdqu32   %zmm7,(448)(%rdi)
vmovdqu32   %zmm8,(512)(%rdi)
vmovdqu32   %zmm9,(576)(%rdi)
vmovdqu32   %zmm10,(640)(%rdi)
vmovdqu32   %zmm11,(704)(%rdi)
vmovdqu32   %zmm12,(768)(%rdi)
vmovdqu32   %zmm13,(832)(%rdi)
vmovdqu32   %zmm14,(896)(%rdi)
vmovdqu32   %zmm15,(960)(%rdi)

ret

.p2align 5   //instructs the assembler to align the following instruction or data on a boundary that is a power of 2 and here is equal to 2^5, or 32 bytes
.global cdecl(addmask_asm)
cdecl(addmask_asm):

vpbroadcastd		_16xmasks(%rip),%zmm31

//load answer[0]-answer[239]
vmovdqu32   (%rdi),%zmm0
vmovdqu32   (64)(%rdi),%zmm1
vmovdqu32   (128)(%rdi),%zmm2
vmovdqu32   (192)(%rdi),%zmm3
vmovdqu32   (256)(%rdi),%zmm4
vmovdqu32   (320)(%rdi),%zmm5
vmovdqu32   (384)(%rdi),%zmm6
vmovdqu32   (448)(%rdi),%zmm7
vmovdqu32   (512)(%rdi),%zmm8
vmovdqu32   (576)(%rdi),%zmm9
vmovdqu32   (640)(%rdi),%zmm10
vmovdqu32   (704)(%rdi),%zmm11
vmovdqu32   (768)(%rdi),%zmm12
vmovdqu32   (832)(%rdi),%zmm13
vmovdqu32   (896)(%rdi),%zmm14

//load s_table[N-i]-s_table[239+N+i]
vmovdqu32   (%rsi),%zmm16
vmovdqu32   (64)(%rsi),%zmm17
vmovdqu32   (128)(%rsi),%zmm18
vmovdqu32   (192)(%rsi),%zmm19
vmovdqu32   (256)(%rsi),%zmm20
vmovdqu32   (320)(%rsi),%zmm21
vmovdqu32   (384)(%rsi),%zmm22
vmovdqu32   (448)(%rsi),%zmm23
vmovdqu32   (512)(%rsi),%zmm24
vmovdqu32   (576)(%rsi),%zmm25
vmovdqu32   (640)(%rsi),%zmm26
vmovdqu32   (704)(%rsi),%zmm27
vmovdqu32   (768)(%rsi),%zmm28
vmovdqu32   (832)(%rsi),%zmm29
vmovdqu32   (896)(%rsi),%zmm30
//gama-s_table
vpsubd    %zmm16,%zmm31,%zmm16
vpsubd    %zmm17,%zmm31,%zmm17
vpsubd    %zmm18,%zmm31,%zmm18
vpsubd    %zmm19,%zmm31,%zmm19
vpsubd    %zmm20,%zmm31,%zmm20
vpsubd    %zmm21,%zmm31,%zmm21
vpsubd    %zmm22,%zmm31,%zmm22
vpsubd    %zmm23,%zmm31,%zmm23
vpsubd    %zmm24,%zmm31,%zmm24
vpsubd    %zmm25,%zmm31,%zmm25
vpsubd    %zmm26,%zmm31,%zmm26
vpsubd    %zmm27,%zmm31,%zmm27
vpsubd    %zmm28,%zmm31,%zmm28
vpsubd    %zmm29,%zmm31,%zmm29
vpsubd    %zmm30,%zmm31,%zmm30
//add
vpaddd       %zmm0,%zmm16,%zmm0
vpaddd       %zmm1,%zmm17,%zmm1
vpaddd       %zmm2,%zmm18,%zmm2
vpaddd       %zmm3,%zmm19,%zmm3
vpaddd       %zmm4,%zmm20,%zmm4
vpaddd       %zmm5,%zmm21,%zmm5
vpaddd       %zmm6,%zmm22,%zmm6
vpaddd       %zmm7,%zmm23,%zmm7
vpaddd       %zmm8,%zmm24,%zmm8
vpaddd       %zmm9,%zmm25,%zmm9
vpaddd       %zmm10,%zmm26,%zmm10
vpaddd       %zmm11,%zmm27,%zmm11
vpaddd       %zmm12,%zmm28,%zmm12
vpaddd       %zmm13,%zmm29,%zmm13
vpaddd       %zmm14,%zmm30,%zmm14
//store answer[0]-answer[239]
vmovdqu32   %zmm0,(0)(%rdi)
vmovdqu32   %zmm1,(64)(%rdi)
vmovdqu32   %zmm2,(128)(%rdi)
vmovdqu32   %zmm3,(192)(%rdi)
vmovdqu32   %zmm4,(256)(%rdi)
vmovdqu32   %zmm5,(320)(%rdi)
vmovdqu32   %zmm6,(384)(%rdi)
vmovdqu32   %zmm7,(448)(%rdi)
vmovdqu32   %zmm8,(512)(%rdi)
vmovdqu32   %zmm9,(576)(%rdi)
vmovdqu32   %zmm10,(640)(%rdi)
vmovdqu32   %zmm11,(704)(%rdi)
vmovdqu32   %zmm12,(768)(%rdi)
vmovdqu32   %zmm13,(832)(%rdi)
vmovdqu32   %zmm14,(896)(%rdi)


//load answer[240]-answer[255]
vmovdqu32   (960)(%rdi),%zmm0
//load s_table[240N-i]-s_table[255+N+i]
vmovdqu32   (960)(%rsi),%zmm16
vpsubd      %zmm16,%zmm31,%zmm16
vpaddd      %zmm0,%zmm16,%zmm0
vmovdqu32   %zmm0,(960)(%rdi)
ret


.p2align 5   //instructs the assembler to align the following instruction or data on a boundary that is a power of 2 and here is equal to 2^5, or 32 bytes
.global cdecl(addmask_asm_prime)
cdecl(addmask_asm_prime):

vpbroadcastd		_16xmask2(%rip),%zmm31

//load answer[0]-answer[239]
vmovdqu32   (%rdi),%zmm0
vmovdqu32   (64)(%rdi),%zmm1
vmovdqu32   (128)(%rdi),%zmm2
vmovdqu32   (192)(%rdi),%zmm3
vmovdqu32   (256)(%rdi),%zmm4
vmovdqu32   (320)(%rdi),%zmm5
vmovdqu32   (384)(%rdi),%zmm6
vmovdqu32   (448)(%rdi),%zmm7
vmovdqu32   (512)(%rdi),%zmm8
vmovdqu32   (576)(%rdi),%zmm9
vmovdqu32   (640)(%rdi),%zmm10
vmovdqu32   (704)(%rdi),%zmm11
vmovdqu32   (768)(%rdi),%zmm12
vmovdqu32   (832)(%rdi),%zmm13
vmovdqu32   (896)(%rdi),%zmm14

//load s_table[N-i]-s_table[239+N+i]
vmovdqu32   (%rsi),%zmm16
vmovdqu32   (64)(%rsi),%zmm17
vmovdqu32   (128)(%rsi),%zmm18
vmovdqu32   (192)(%rsi),%zmm19
vmovdqu32   (256)(%rsi),%zmm20
vmovdqu32   (320)(%rsi),%zmm21
vmovdqu32   (384)(%rsi),%zmm22
vmovdqu32   (448)(%rsi),%zmm23
vmovdqu32   (512)(%rsi),%zmm24
vmovdqu32   (576)(%rsi),%zmm25
vmovdqu32   (640)(%rsi),%zmm26
vmovdqu32   (704)(%rsi),%zmm27
vmovdqu32   (768)(%rsi),%zmm28
vmovdqu32   (832)(%rsi),%zmm29
vmovdqu32   (896)(%rsi),%zmm30
//gama-s_table
vpsubd    %zmm16,%zmm31,%zmm16
vpsubd    %zmm17,%zmm31,%zmm17
vpsubd    %zmm18,%zmm31,%zmm18
vpsubd    %zmm19,%zmm31,%zmm19
vpsubd    %zmm20,%zmm31,%zmm20
vpsubd    %zmm21,%zmm31,%zmm21
vpsubd    %zmm22,%zmm31,%zmm22
vpsubd    %zmm23,%zmm31,%zmm23
vpsubd    %zmm24,%zmm31,%zmm24
vpsubd    %zmm25,%zmm31,%zmm25
vpsubd    %zmm26,%zmm31,%zmm26
vpsubd    %zmm27,%zmm31,%zmm27
vpsubd    %zmm28,%zmm31,%zmm28
vpsubd    %zmm29,%zmm31,%zmm29
vpsubd    %zmm30,%zmm31,%zmm30
//add
vpaddd       %zmm0,%zmm16,%zmm0
vpaddd       %zmm1,%zmm17,%zmm1
vpaddd       %zmm2,%zmm18,%zmm2
vpaddd       %zmm3,%zmm19,%zmm3
vpaddd       %zmm4,%zmm20,%zmm4
vpaddd       %zmm5,%zmm21,%zmm5
vpaddd       %zmm6,%zmm22,%zmm6
vpaddd       %zmm7,%zmm23,%zmm7
vpaddd       %zmm8,%zmm24,%zmm8
vpaddd       %zmm9,%zmm25,%zmm9
vpaddd       %zmm10,%zmm26,%zmm10
vpaddd       %zmm11,%zmm27,%zmm11
vpaddd       %zmm12,%zmm28,%zmm12
vpaddd       %zmm13,%zmm29,%zmm13
vpaddd       %zmm14,%zmm30,%zmm14
//store answer[0]-answer[239]
vmovdqu32   %zmm0,(0)(%rdi)
vmovdqu32   %zmm1,(64)(%rdi)
vmovdqu32   %zmm2,(128)(%rdi)
vmovdqu32   %zmm3,(192)(%rdi)
vmovdqu32   %zmm4,(256)(%rdi)
vmovdqu32   %zmm5,(320)(%rdi)
vmovdqu32   %zmm6,(384)(%rdi)
vmovdqu32   %zmm7,(448)(%rdi)
vmovdqu32   %zmm8,(512)(%rdi)
vmovdqu32   %zmm9,(576)(%rdi)
vmovdqu32   %zmm10,(640)(%rdi)
vmovdqu32   %zmm11,(704)(%rdi)
vmovdqu32   %zmm12,(768)(%rdi)
vmovdqu32   %zmm13,(832)(%rdi)
vmovdqu32   %zmm14,(896)(%rdi)


//load answer[240]-answer[255]
vmovdqu32   (960)(%rdi),%zmm0
//load s_table[240N-i]-s_table[255+N+i]
vmovdqu32   (960)(%rsi),%zmm16
vpsubd      %zmm16,%zmm31,%zmm16
vpaddd      %zmm0,%zmm16,%zmm0
vmovdqu32   %zmm0,(960)(%rdi)
ret