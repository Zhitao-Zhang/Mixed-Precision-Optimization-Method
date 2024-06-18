#include <stdio.h>
#include "cuda_fp16.h"
#include "cuda_runtime.h"

typedef half data_t;
typedef struct
{
    //------------------------------------------------------------------------------------------------------------------
    // host
    data_t *h_vux, *h_vuy, *h_vuz;
    data_t *h_vwx, *h_vwy, *h_vwz;
    data_t *h_txx, *h_txy, *h_txz, *h_tyy, *h_tyz, *h_tzz, *h_ss;
    data_t *h_VelocityWParameter1x, *h_VelocityWParameter1y, *h_VelocityWParameter1z;
    data_t *h_VelocityWParameter2x, *h_VelocityWParameter2y, *h_VelocityWParameter2z;
    data_t *h_VelocityWParameter3x, *h_VelocityWParameter3y, *h_VelocityWParameter3z;
    data_t *h_VelocityUParameter1x, *h_VelocityUParameter1y, *h_VelocityUParameter1z;
    data_t *h_VelocityUParameter2x, *h_VelocityUParameter2y, *h_VelocityUParameter2z;
    data_t *h_PressParameter1, *h_PressParameter2;
    data_t *h_StressParameter1, *h_StressParameter2, *h_StressParameter3;
    data_t *h_StressParameterxy, *h_StressParameterxz, *h_StressParameteryz;
    data_t *h_pmlxSxx, *h_pmlySxy, *h_pmlzSxz, *h_pmlxSxy, *h_pmlySyy, *h_pmlzSyz, *h_pmlxSxz, *h_pmlySyz, *h_pmlzSzz;
    data_t *h_pmlxVux, *h_pmlyVuy, *h_pmlzVuz, *h_pmlxVuy, *h_pmlyVux, *h_pmlzVux, *h_pmlxVuz, *h_pmlyVuz, *h_pmlzVuy, *h_pmlxVwx, *h_pmlyVwy, *h_pmlzVwz, *h_pmlxss, *h_pmlyss, *h_pmlzss;
    data_t *h_SXxx, *h_SXxy, *h_SXxz, *h_SYxy, *h_SYyy, *h_SYyz, *h_SZxz, *h_SZyz, *h_SZzz, *h_SXss, *h_SYss, *h_SZss;
    data_t *h_Vuxx, *h_Vuyy, *h_Vuzz, *h_Vuxy, *h_Vuyx, *h_Vuyz, *h_Vuzy, *h_Vuxz, *h_Vuzx, *h_Vwxx, *h_Vwyy, *h_Vwzz;
    data_t *h_vwx2, *h_vwy2, *h_vwz2;
    data_t *h_dxi, *h_dyj, *h_dzk, *h_dxi2, *h_dyj2, *h_dzk2, *h_e_dxi, *h_e_dyj, *h_e_dzk, *h_e_dxi2, *h_e_dyj2, *h_e_dzk2;
    //------------------------------------------------------------------------------------------------------------------
    // device
    data_t *d_vux, *d_vuy, *d_vuz;
    data_t *d_vwx, *d_vwy, *d_vwz;
    data_t *d_txx, *d_txy, *d_txz, *d_tyy, *d_tyz, *d_tzz, *d_ss;
    data_t *d_VelocityWParameter1x, *d_VelocityWParameter1y, *d_VelocityWParameter1z;
    data_t *d_VelocityWParameter2x, *d_VelocityWParameter2y, *d_VelocityWParameter2z;
    data_t *d_VelocityWParameter3x, *d_VelocityWParameter3y, *d_VelocityWParameter3z;
    data_t *d_VelocityUParameter1x, *d_VelocityUParameter1y, *d_VelocityUParameter1z;
    data_t *d_VelocityUParameter2x, *d_VelocityUParameter2y, *d_VelocityUParameter2z;
    data_t *d_PressParameter1, *d_PressParameter2;
    data_t *d_StressParameter1, *d_StressParameter2, *d_StressParameter3;
    data_t *d_StressParameterxy, *d_StressParameterxz, *d_StressParameteryz;
    data_t *d_pmlxSxx, *d_pmlySxy, *d_pmlzSxz, *d_pmlxSxy, *d_pmlySyy, *d_pmlzSyz, *d_pmlxSxz, *d_pmlySyz, *d_pmlzSzz;
    data_t *d_pmlxVux, *d_pmlyVuy, *d_pmlzVuz, *d_pmlxVuy, *d_pmlyVux, *d_pmlzVux, *d_pmlxVuz, *d_pmlyVuz, *d_pmlzVuy, *d_pmlxVwx, *d_pmlyVwy, *d_pmlzVwz, *d_pmlxss, *d_pmlyss, *d_pmlzss;
    data_t *d_SXxx, *d_SXxy, *d_SXxz, *d_SYxy, *d_SYyy, *d_SYyz, *d_SZxz, *d_SZyz, *d_SZzz, *d_SXss, *d_SYss, *d_SZss;
    data_t *d_Vuxx, *d_Vuyy, *d_Vuzz, *d_Vuxy, *d_Vuyx, *d_Vuyz, *d_Vuzy, *d_Vuxz, *d_Vuzx, *d_Vwxx, *d_Vwyy, *d_Vwzz;
    data_t *d_vwx2, *d_vwy2, *d_vwz2;
    data_t *d_dxi, *d_dyj, *d_dzk, *d_dxi2, *d_dyj2, *d_dzk2, *d_e_dxi, *d_e_dyj, *d_e_dzk, *d_e_dxi2, *d_e_dyj2, *d_e_dzk2;
    //------------------------------------------------------------------------------------------------------------------
    // halo区打包
    data_t *halo_Vux_down_pack, *halo_Vuy_down_pack, *halo_Vuz_up_pack, *halo_Vwz_up_pack;
    data_t *halo_Tzz_down_pack, *halo_Txz_up_pack, *halo_Tyz_up_pack, *halo_ss_down_pack;
    // top
    data_t *halo_Vux_right_top_pack, *halo_Vux_back_top_pack;
    data_t *halo_Vuy_left_top_pack, *halo_Vuy_front_top_pack;
    data_t *halo_Vuz_left_top_pack, *halo_Vuz_back_top_pack;
    data_t *halo_Vwx_right_top_pack;
    data_t *halo_Vwy_front_top_pack;

    data_t *halo_Txx_left_top_pack;
    data_t *halo_Txy_right_top_pack, *halo_Txy_front_top_pack;
    data_t *halo_Txz_right_top_pack;
    data_t *halo_Tyy_back_top_pack;
    data_t *halo_Tyz_front_top_pack;
    data_t *halo_ss_left_top_pack, *halo_ss_back_top_pack;
    // mid
    data_t *halo_Vux_right_mid_pack, *halo_Vux_back_mid_pack;
    data_t *halo_Vuy_left_mid_pack, *halo_Vuy_front_mid_pack;
    data_t *halo_Vuz_left_mid_pack, *halo_Vuz_back_mid_pack;
    data_t *halo_Vwx_right_mid_pack;
    data_t *halo_Vwy_front_mid_pack;

    data_t *halo_Txx_left_mid_pack;
    data_t *halo_Txy_right_mid_pack, *halo_Txy_front_mid_pack;
    data_t *halo_Txz_right_mid_pack;
    data_t *halo_Tyy_back_mid_pack;
    data_t *halo_Tyz_front_mid_pack;
    data_t *halo_ss_left_mid_pack, *halo_ss_back_mid_pack;
    // bottom
    data_t *halo_Vux_right_bottom_pack, *halo_Vux_back_bottom_pack;
    data_t *halo_Vuy_left_bottom_pack, *halo_Vuy_front_bottom_pack;
    data_t *halo_Vuz_left_bottom_pack, *halo_Vuz_back_bottom_pack;
    data_t *halo_Vwx_right_bottom_pack;
    data_t *halo_Vwy_front_bottom_pack;

    data_t *halo_Txx_left_bottom_pack;
    data_t *halo_Txy_right_bottom_pack, *halo_Txy_front_bottom_pack;
    data_t *halo_Txz_right_bottom_pack;
    data_t *halo_Tyy_back_bottom_pack;
    data_t *halo_Tyz_front_bottom_pack;
    data_t *halo_ss_left_bottom_pack, *halo_ss_back_bottom_pack;
    //------------------------------------------------------------------------------------------------------------------
    // halo区接收
    data_t *halo_Vwz_up_recv, *halo_Vuz_up_recv, *halo_Txz_up_recv, *halo_Tyz_up_recv;
    data_t *halo_Tzz_down_recv, *halo_Vux_down_recv, *halo_Vuy_down_recv, *halo_ss_down_recv;
    // top
    data_t *halo_Vux_right_top_recv, *halo_Vux_back_top_recv;
    data_t *halo_Vuy_left_top_recv, *halo_Vuy_front_top_recv;
    data_t *halo_Vuz_left_top_recv, *halo_Vuz_back_top_recv;
    data_t *halo_Vwx_right_top_recv;
    data_t *halo_Vwy_front_top_recv;

    data_t *halo_Txx_left_top_recv;
    data_t *halo_Txy_right_top_recv, *halo_Txy_front_top_recv;
    data_t *halo_Txz_right_top_recv;
    data_t *halo_Tyy_back_top_recv;
    data_t *halo_Tyz_front_top_recv;
    data_t *halo_ss_left_top_recv, *halo_ss_back_top_recv;
    // mid
    data_t *halo_Vux_right_mid_recv, *halo_Vux_back_mid_recv;
    data_t *halo_Vuy_left_mid_recv, *halo_Vuy_front_mid_recv;
    data_t *halo_Vuz_left_mid_recv, *halo_Vuz_back_mid_recv;
    data_t *halo_Vwx_right_mid_recv;
    data_t *halo_Vwy_front_mid_recv;

    data_t *halo_Txx_left_mid_recv;
    data_t *halo_Txy_right_mid_recv, *halo_Txy_front_mid_recv;
    data_t *halo_Txz_right_mid_recv;
    data_t *halo_Tyy_back_mid_recv;
    data_t *halo_Tyz_front_mid_recv;
    data_t *halo_ss_left_mid_recv, *halo_ss_back_mid_recv;
    // bottom
    data_t *halo_Vux_right_bottom_recv, *halo_Vux_back_bottom_recv;
    data_t *halo_Vuy_left_bottom_recv, *halo_Vuy_front_bottom_recv;
    data_t *halo_Vuz_left_bottom_recv, *halo_Vuz_back_bottom_recv;
    data_t *halo_Vwx_right_bottom_recv;
    data_t *halo_Vwy_front_bottom_recv;

    data_t *halo_Txx_left_bottom_recv;
    data_t *halo_Txy_right_bottom_recv, *halo_Txy_front_bottom_recv;
    data_t *halo_Txz_right_bottom_recv;
    data_t *halo_Tyy_back_bottom_recv;
    data_t *halo_Tyz_front_bottom_recv;
    data_t *halo_ss_left_bottom_recv, *halo_ss_back_bottom_recv;

} blockInfo;

void callCUDAKernel();