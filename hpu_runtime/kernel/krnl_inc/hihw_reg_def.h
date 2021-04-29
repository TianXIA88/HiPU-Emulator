#ifndef HIPU200CLIB_RV_REG_DEF_H__
#define HIPU200CLIB_RV_REG_DEF_H__

#define sr0                     (0)
#define sr1                     (1)
#define sr2                     (2)
#define sr3                     (3)
#define sr4                     (4)
#define sr5                     (5)
#define sr6                     (6)
#define sr7                     (7)
#define sr8                     (8)
#define sr9                     (9)
#define sr10                    (10)
#define sr11                    (11)
#define sr12                    (12)
#define sr13                    (13)
#define sr14                    (14)
#define sr15                    (15)
#define sr16                    (16)
#define sr17                    (17)
#define sr18                    (18)
#define sr19                    (19)
#define sr20                    (20)
#define sr21                    (21)
#define sr22                    (22)
#define sr23                    (23)
#define sr24                    (24)
#define sr25                    (25)
#define sr26                    (26)
#define sr27                    (27)
#define sr28                    (28)
#define sr29                    (29)
#define sr30                    (30)
#define sr31                    (31)

#define sr_zero                 (sr0)  //hardware zero
#define sr_ra                   (sr1)  //return addr
#define sr_sp                   (sr2)  //stack pointer
#define sr_gp                   (sr3)  //global pointer
#define sr_tp                   (sr4)  //thread pointer
#define sr_t0                   (sr5)  //temporary
#define sr_t1                   (sr6)  //temporary
#define sr_t2                   (sr7)  //temporary
#define sr_s0                   (sr8)  //saved reg
#define sr_s1                   (sr9)  //saved reg
#define sr_a0                   (sr10) //function argument, return val
#define sr_a1                   (sr11) //function argument, return val
#define sr_a2                   (sr12) //function argument
#define sr_a3                   (sr13) //function argument
#define sr_a4                   (sr14) //function argument
#define sr_a5                   (sr15) //function argument
#define sr_a6                   (sr16) //function argument
#define sr_a7                   (sr17) //function argument
#define sr_s2                   (sr18) //saved reg
#define sr_s3                   (sr19) //saved reg
#define sr_s4                   (sr20) //saved reg
#define sr_s5                   (sr21) //saved reg
#define sr_s6                   (sr22) //saved reg
#define sr_s7                   (sr23) //saved reg
#define sr_s8                   (sr24) //saved reg
#define sr_s9                   (sr25) //saved reg
#define sr_s10                  (sr26) //saved reg
#define sr_s11                  (sr27) //saved reg
#define sr_t3                   (sr28) //temporary
#define sr_t4                   (sr29) //temporary
#define sr_t5                   (sr30) //temporary
#define sr_t6                   (sr31) //temporary

// Alias
#define zero                    (sr0) 
#define ra                      (sr1) 
#define sp                      (sr2) 
#define gp                      (sr3) 
#define fp                      (sr8)
#define tp                      (sr4) 
#define t0                      (sr5) 
#define t1                      (sr6) 
#define t2                      (sr7) 
#define t3                      (sr28) 
#define t4                      (sr29) 
#define t5                      (sr30) 
#define t6                      (sr31)
#define s0                      (sr8) 
#define s1                      (sr9) 
#define s2                      (sr18) 
#define s3                      (sr19) 
#define s4                      (sr20) 
#define s5                      (sr21)
#define s6                      (sr22) 
#define s7                      (sr23) 
#define s8                      (sr24) 
#define s9                      (sr25) 
#define s10                     (sr26) 
#define s11                     (sr27)
#define a0                      (sr10) 
#define a1                      (sr11) 
#define a2                      (sr12) 
#define a3                      (sr13) 
#define a4                      (sr14) 
#define a5                      (sr15) 
#define a6                      (sr16) 
#define a7                      (sr17)

#define vr0                     (0)
#define vr1                     (1)
#define vr2                     (2)
#define vr3                     (3)
#define vr4                     (4)
#define vr5                     (5)
#define vr6                     (6)
#define vr7                     (7)
#define vr8                     (8)
#define vr9                     (9)
#define vr10                    (10)
#define vr11                    (11)
#define vr12                    (12)
#define vr13                    (13)
#define vr14                    (14)
#define vr15                    (15)

#define vpr0                    (0)
#define vpr1                    (1)
#define vpr2                    (2)
#define vpr3                    (3)
#define vpr4                    (4)
#define vpr5                    (5)
#define vpr6                    (6)
#define vpr7                    (7)

#endif /*HIPU200CLIB_RV_REG_DEF_H__*/
