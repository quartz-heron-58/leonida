; Mesh shader (NGG, wave64, 128 threads = 2 waves): LDS that only wave 0 stores to and both waves read, in the two
; places a pipeline-library mesh program has it.
;   Opening lines: wave 0 (s3[27:24] == 0, tested with s_and_b32 / s_cbranch_scc1) stores 60 at LDS[8] from lane 0;
;   both waves pass an s_barrier and read LDS[8].
;   Before the exports: wave 0 (tested with s_bfe_u32 / s_cmp_eq_i32 / s_cbranch_scc0) stores the primitive count 1
;   at LDS[0] and the vertex count 65 at LDS[4] and sends the allocation request; wave 1 passes the same barriers
;   and reads both back.
; A wave that read 60, 1 and 65 sets the blue channel of param0 to 0.5 in the vertices it exports, else 0.
; The triangle is ms_tri's, with its vertices from threads 0 and 1 (wave 0) and thread 64 (wave 1), as in
; ms_w64x_wave_loops: the picture is ms_tri's only when wave 1 read what wave 0 stored. Under --wave64-emulate the
; two waves run side by side and meet at the barriers; with --mesh-serial-waves wave 0 runs first and %lds, which is
; Workgroup memory, still holds its values when wave 1 runs (ms_lds_wave0_only in tools/rdna2spirv.py).
; ms_wave_info_s3 is the other kind: every wave adds to LDS, and it keeps the side-by-side form.
; Needs a DXBC with PSV0 (128 threads, 128/64 outputs, triangles; see scripts/run_gpu_tests.py):
;   VKH_MESH=1 vk_harness ms_w64x_lds_wave0.spv fs_passthru.spv out.raw
;   -> every pixel = ((x+0.5)/64, (y+0.5)/64, 0.5, 1), identical to ms_tri.
; mattr: +wavefrontsize64,-wavefrontsize32
s_mov_b64 exec, 1
s_and_b32 s8, s3, 0xf000000
s_cbranch_scc1 6
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v12, 60
ds_write_b32 v10, v12 offset:8
s_barrier
s_branch 1
s_barrier
v_mov_b32_e32 v10, 0
ds_read_b32 v14, v10 offset:8
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s9, v14
s_mov_b64 exec, -1
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mbcnt_hi_u32_b32 v0, -1, v0
s_bfe_u32 s4, s3, 0x40018
v_lshl_add_u32 v0, s4, 6, v0
s_barrier
s_cmp_eq_i32 s4, 0
s_cbranch_scc0 15
s_mov_b64 exec, 1
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v16, 1
v_mov_b32_e32 v17, 0x41
ds_write_b32 v10, v16
ds_write_b32 v10, v17 offset:4
s_mov_b64 exec, -1
s_barrier
s_mov_b32 m0, 0x1041
s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
s_branch 1
s_barrier
s_mov_b64 exec, 1
v_mov_b32_e32 v10, 0
ds_read_b32 v18, v10
ds_read_b32 v19, v10 offset:4
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s21, v18
v_readfirstlane_b32 s10, v19
s_mov_b64 exec, -1
s_cmp_eq_u32 s9, 60
s_cselect_b32 s13, 1, 0
s_cmp_eq_u32 s21, 1
s_cselect_b32 s13, s13, 0
s_cmp_eq_u32 s10, 0x41
s_cselect_b32 s13, s13, 0
v_cvt_f32_u32_e32 v15, s13
v_mul_f32_e32 v15, 0.5, v15
v_cmp_eq_u32_e32 vcc, 1, v0
v_cndmask_b32_e64 v1, -1.0, 0x40400000, vcc
v_cndmask_b32_e64 v3, 0, 2.0, vcc
v_cmp_eq_u32_e32 vcc, 64, v0
v_cndmask_b32_e64 v2, -1.0, 0x40400000, vcc
v_cndmask_b32_e64 v4, 0, 2.0, vcc
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 1.0
v_mov_b32_e32 v8, v15
v_cmp_gt_u32_e64 s[16:17], 2, v0
v_cmp_eq_u32_e64 s[18:19], 64, v0
s_or_b64 exec, s[16:17], s[18:19]
exp pos0 v1, v2, v6, v7 done
exp param0 v3, v4, v8, v7
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v9, 0x4000400
exp prim v9, off, off, off done
s_endpgm
