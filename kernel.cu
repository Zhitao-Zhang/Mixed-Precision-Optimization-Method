#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include "common.h"
#include <mpi.h>
#include <iostream>
#include "outputfunc.h"
#define PI 3.1415926535
#define e 2.718281828
#define NZ 300
#define NY 180 
#define NX 180
#define NP 30
#define NL 200
#define m 5    
#define MM 1    
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 1
//half parameter
#define d_Cvwp1 1.0
#define d_Cvwp2 1.0
#define d_Cvwp3 1.0
#define d_Cvup1 1.0
#define d_Cvup2 1.0
#define d_Cpp1 0.01
#define d_Cpp2 0.01
#define d_Csp1 0.01
#define d_Csp2 0.01
#define d_Csp3 0.01
#define d_Csp4 0.01
#define d_Cpmll 100.0
#define PX 3
#define PY 3
#define PZ 3
#define tran 1
#define local_x (NX + 2 * NP) / PX
#define local_y (NY + 2 * NP) / PY
#define local_z (NZ + 2 * NP) / PZ / 3
#define PI 3.1415926535 
#define MAX_REQUESTS 100
#define CUDA_CHECK(call)                                                         \
	do                                                                           \
	{                                                                            \
		cudaError_t result = call;                                               \
		if (result != cudaSuccess)                                               \
		{                                                                        \
			std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
					  << cudaGetErrorString(result) << std::endl;                \
			exit(EXIT_FAILURE);                                                  \
		}                                                                        \
	} while (0)
//------------------------------------------------------------------------------------------------------------------
//pack and unpack halo
__global__ void gather_halo_yz(data_t *model, data_t *right_halo, size_t x, int part){

	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	//model濞戞挸锕�?▓鎲僨fset
	int base_offset = x;
	int halo_offset = iy + iz * local_y;
	int model_offset = (iz + part * local_z) * local_x * local_y + iy * local_x + base_offset;
	right_halo[halo_offset] = model[model_offset]; 
	
}

__global__ void gather_halo_xz(data_t *model, data_t *front_halo, size_t y, int part){

	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	//model濞戞挸锕�?▓鎲僨fset
	int base_offset = y * local_x;
	int halo_offset = iz * local_x  + ix;
	int model_offset = (iz + part * local_z) * local_x * local_y + ix + base_offset;
	front_halo[halo_offset] = model[model_offset]; 
}

__global__ void gather_halo_xy(data_t *model, data_t *up_halo, size_t z, int part){

	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	//model濞戞挸锕�?▓鎲僨fset
	//local_x闁圭?娲ㄥ▓鎴�?及椤栨碍�???☉鎾?亾濞戞挾灏爌i閺夆晜绋撻埢鍏肩▔閻′線�???悷鐗堝€婚柣銊ュ?閵囧洨浜歌箛瀣у亾閿燂�??
	int base_offset = local_x * local_y * (z + part * local_z);
	int halo_offset = ix + iy * local_x;
	int model_offset = base_offset + halo_offset;
	up_halo[halo_offset] = model[model_offset]; 

}

__global__ void scatter_halo_yz(data_t *model, data_t *right_halo, size_t x, int part){

	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;

	//model濞戞挸锕�?▓鎲僨fset
	int base_offset = x;
	int halo_offset = iy + iz * local_y;
	int model_offset = (iz + part * local_z) * local_x * local_y + iy * local_x + base_offset;
	model[model_offset] = right_halo[halo_offset]; 

}

__global__ void scatter_halo_xz(data_t *model, data_t *front_halo, size_t y, int part){

	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;

	//model濞戞挸锕�?▓鎲僨fset
	int base_offset = y * local_x;

	int halo_offset = iz * local_x  + ix;
	int model_offset = (iz + part * local_z) * local_x * local_y + ix + base_offset;
	model[model_offset] = front_halo[halo_offset];

}

__global__ void scatter_halo_xy(data_t *model, data_t *up_halo, size_t z, int part){

	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	int base_offset = local_x * local_y * (z + part * local_z);
	int halo_offset = ix + iy * local_x;
	int model_offset = base_offset + halo_offset;
	model[model_offset] = up_halo[halo_offset]; 
}
//------------------------------------------------------------------------------------------------------------------
//computer Source
__global__ void Source(data_t *d_txx, data_t *d_tyy, data_t *d_tzz, float I_sou, int sn, int nxt, int nyt, int nzt) 
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = ix + iy * nxt + iz * nxt * nyt;
	if (offset == sn)
	{
		d_txx[offset] = (float)d_txx[offset] + I_sou;
		d_tyy[offset] = (float)d_tyy[offset] + I_sou;
		d_tzz[offset] = (float)d_tzz[offset] + I_sou;
	}
}
//------------------------------------------------------------------------------------------------------------------
//computer velocity
__global__ void FD_Vtop(data_t* vux, data_t* vuy, data_t* vuz,
	data_t* txx, data_t* tyy, data_t* tzz, data_t* txz, data_t* txy, data_t* tyz,
	data_t* pmlxSxx, data_t* pmlySxy, data_t* pmlzSxz, data_t* pmlxSxy, data_t* pmlySyy, data_t* pmlzSyz, data_t* pmlxSxz, data_t* pmlySyz, data_t* pmlzSzz,
	data_t* SXxx, data_t* SXxy, data_t* SXxz, data_t* SYxy, data_t* SYyy, data_t* SYyz, data_t* SZxz, data_t* SZyz, data_t* SZzz,
	data_t* e_dxi, data_t* dxi, data_t* e_dxi2, data_t* dxi2, data_t* e_dyj, data_t* dyj, data_t* e_dyj2, data_t* dyj2, data_t* e_dzk, data_t* dzk, data_t* dzk2, data_t* e_dzk2,
	data_t* ss, data_t* vwx, data_t* vwy, data_t* vwz, data_t* vwx2, data_t* vwy2, data_t* vwz2, data_t* SXss, data_t* SYss, data_t* SZss, data_t* pmlxss, data_t* pmlyss, data_t* pmlzss,
	data_t* VelocityWParameter1x, data_t* VelocityWParameter1y, data_t* VelocityWParameter1z, data_t* VelocityWParameter2x, data_t* VelocityWParameter2y, data_t* VelocityWParameter2z, data_t* VelocityWParameter3x, data_t* VelocityWParameter3y, data_t* VelocityWParameter3z,
	data_t* VelocityUParameter1x, data_t* VelocityUParameter1y, data_t* VelocityUParameter1z, data_t* VelocityUParameter2x, data_t* VelocityUParameter2y, data_t* VelocityUParameter2z, float DT, int nxt, int nyt, int nzt, int partIndex)
{
	float x1, x2, x3;
	float z1, z2, z3;
	float y1, y2, y3;
	float s1, s2, s3;
	float H = 100;
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = ix + iy * nxt + (iz + partIndex * nzt) * nxt * nyt;
	int offset_b = ix + iy * nxt + ((iz + partIndex * nzt) - 1) * nxt * nyt;//婵炴埊鎷??
	int offset_r = ix + 1 + iy * nxt + (iz + partIndex * nzt) * nxt * nyt;//闂侀潻鎷??
	int offset_h = ix + (iy - 1) * nxt + (iz + partIndex * nzt) * nxt * nyt;//闂侀潻鎷??
	int offset_q = ix + (iy + 1) * nxt + (iz + partIndex * nzt) * nxt * nyt;//闂侀潻鎷??
	int offset_l = ix - 1 + iy * nxt + (iz + partIndex * nzt) * nxt * nyt;//閻庣櫢鎷??
	int offset_u = ix + iy * nxt + (1 + (iz + partIndex * nzt)) * nxt * nyt;//婵炴埊鎷??

	if(ix > 0 && iy > 0 && iz >= 0 && ix < (nxt - 1) && iy < (nyt - 1) && iz < (nzt - 1))
	{
		x1 = ((float)txx[offset_r] - (float)txx[offset]) * H;
		x2 = ((float)txy[offset] - (float)txy[offset_h]) * H;
		x3 = ((float)txz[offset] - (float)txz[offset_b]) * H;
		s1 = ((float)ss[offset_r] - (float)ss[offset]) * H;

		y1 = ((float)tyy[offset_q] - (float)tyy[offset]) * H;
		y2 = ((float)txy[offset] - (float)txy[offset_l]) * H;
		y3 = ((float)tyz[offset] - (float)tyz[offset_b]) * H;
		s2 = ((float)ss[offset_q] - (float)ss[offset]) * H;

		z1 = ((float)tzz[offset_u] - (float)tzz[offset]) * H;
		z2 = ((float)txz[offset] - (float)txz[offset_l]) * H;
		z3 = ((float)tyz[offset] - (float)tyz[offset_h]) * H;
		s3 = ((float)ss[offset_u] - (float)ss[offset]) * H;


		pmlxSxx[offset] = (float)pmlxSxx[offset] * (float)e_dxi2[offset] * d_Cpmll + (-DT * (float)dxi2[offset] * d_Cpmll * 0.5) * ((float)e_dxi2[offset] * d_Cpmll * (float)SXxx[offset] + x1);
		pmlySxy[offset] = (float)pmlySxy[offset] * (float)e_dyj[offset] * d_Cpmll + (-DT * (float)dyj[offset] * d_Cpmll * 0.5) * ((float)e_dyj[offset] * d_Cpmll * (float)SXxy[offset] + x2);
		pmlzSxz[offset] = (float)pmlzSxz[offset] * (float)e_dzk[offset] * d_Cpmll + (-DT * (float)dzk[offset] * d_Cpmll * 0.5) * ((float)e_dzk[offset] * d_Cpmll * (float)SXxz[offset] + x3);
		pmlxss[offset] = (float)pmlxss[offset] * (float)e_dxi2[offset] * d_Cpmll + (-DT * (float)dxi2[offset] * d_Cpmll * 0.5) * ((float)e_dxi2[offset] * d_Cpmll * (float)SXss[offset] + s1);
		SXxx[offset] = x1; SXxy[offset] = x2; SXxz[offset] = x3; SXss[offset] = s1;
		x1 = x1 + (float)pmlxSxx[offset];
		x2 = x2 + (float)pmlySxy[offset];
		x3 = x3 + (float)pmlzSxz[offset];
		s1 = s1 + (float)pmlxss[offset];

		pmlxSxy[offset] = (float)pmlxSxy[offset] * (float)e_dxi[offset] * d_Cpmll + (-DT * (float)dxi[offset] * d_Cpmll * 0.5) * ((float)e_dxi[offset]  * d_Cpmll* (float)SYxy[offset] + y2);
		pmlySyy[offset] = (float)pmlySyy[offset] * (float)e_dyj2[offset] * d_Cpmll + (-DT * (float)dyj2[offset]  * d_Cpmll* 0.5) * ((float)e_dyj2[offset]  * d_Cpmll* (float)SYyy[offset] + y1);
		pmlzSyz[offset] = (float)pmlzSyz[offset] * (float)e_dzk[offset] * d_Cpmll + (-DT * (float)dzk[offset] * d_Cpmll * 0.5) * ((float)e_dzk[offset] * d_Cpmll * (float)SYyz[offset] + y3);
		pmlyss[offset] = (float)pmlyss[offset] * (float)e_dyj2[offset] * d_Cpmll + (-DT * (float)dyj2[offset] * d_Cpmll * 0.5) * ((float)e_dyj2[offset]  * d_Cpmll* (float)SYss[offset] + s2);
		SYxy[offset] = y2; SYyy[offset] = y1; SYyz[offset] = y3; SYss[offset] = s2;
		y2 = y2 + (float)pmlxSxy[offset];
		y1 = y1 + (float)pmlySyy[offset];
		y3 = y3 + (float)pmlzSyz[offset];
		s2 = s2 + (float)pmlyss[offset];

		pmlxSxz[offset] = (float)pmlxSxz[offset] * (float)e_dxi[offset] * d_Cpmll + (-DT * (float)dxi[offset] * d_Cpmll * 0.5) * ((float)e_dxi[offset]  * d_Cpmll* (float)SZxz[offset] + z2);
		pmlySyz[offset] = (float)pmlySyz[offset] * (float)e_dyj[offset] * d_Cpmll + (-DT * (float)dyj[offset] * d_Cpmll * 0.5) * ((float)e_dyj[offset] * d_Cpmll * (float)SZyz[offset] + z3);
		pmlzSzz[offset] = (float)pmlzSzz[offset] * (float)e_dzk2[offset] * d_Cpmll + (-DT * (float)dzk2[offset] * d_Cpmll * 0.5) * ((float)e_dzk2[offset] * d_Cpmll * (float)SZzz[offset] + z1);
		pmlzss[offset] = (float)pmlzss[offset] * (float)e_dzk2[offset] * d_Cpmll + (-DT * (float)dzk2[offset]  * d_Cpmll* 0.5) * ((float)e_dzk2[offset]  * d_Cpmll* (float)SZss[offset] + s3);
		SZxz[offset] = z2; SZyz[offset] = z3; SZzz[offset] = z1; SZss[offset] = s3;
		z2 = z2 + (float)pmlxSxz[offset];
		z3 = z3 + (float)pmlySyz[offset];
		z1 = z1 + (float)pmlzSzz[offset];
		s3 = s3 + (float)pmlzss[offset];

		vwx[offset] = d_Cvwp1 * (float)VelocityWParameter1x[offset] * (float)vwx[offset] - d_Cvwp2 * (float)VelocityWParameter2x[offset] * (x1 + x2 + x3) - d_Cvwp3 * (float)VelocityWParameter3x[offset] * s1;
		vwy[offset] = d_Cvwp1 * (float)VelocityWParameter1y[offset] * (float)vwy[offset] - d_Cvwp2 * (float)VelocityWParameter2y[offset] * (y1 + y2 + y3) - d_Cvwp3 * (float)VelocityWParameter3y[offset] * s2;
		vwz[offset] = d_Cvwp1 * (float)VelocityWParameter1z[offset] * (float)vwz[offset] - d_Cvwp2 * (float)VelocityWParameter2z[offset] * (z1 + z2 + z3) - d_Cvwp3 * (float)VelocityWParameter3y[offset] * s3;


		vux[offset] = (float)vux[offset] + d_Cvup1 * (float)VelocityUParameter1x[offset] * (x1 + x2 + x3) - d_Cvup2 * (float)VelocityUParameter2x[offset] * ((float)vwx[offset] - (float)vwx2[offset]);
		vuy[offset] = (float)vuy[offset] + d_Cvup1 * (float)VelocityUParameter1y[offset] * (y1 + y2 + y3) - d_Cvup2 * (float)VelocityUParameter2y[offset] * ((float)vwy[offset] - (float)vwy2[offset]);
		vuz[offset] = (float)vuz[offset] + d_Cvup1 * (float)VelocityUParameter1z[offset] * (z1 + z2 + z3) - d_Cvup2 * (float)VelocityUParameter2z[offset] * ((float)vwz[offset] - (float)vwz2[offset]);
		vwx2[offset] = vwx[offset]; vwy2[offset] = vwy[offset]; vwz2[offset] = vwz[offset];

   }
}

__global__ void FD_Vmid(data_t *vux, data_t *vuy, data_t *vuz,
	data_t *txx, data_t *tyy, data_t *tzz, data_t *txz, data_t *txy, data_t *tyz,
	data_t *pmlxSxx, data_t *pmlySxy, data_t *pmlzSxz, data_t *pmlxSxy, data_t *pmlySyy, data_t *pmlzSyz, data_t *pmlxSxz, data_t *pmlySyz, data_t *pmlzSzz,
	data_t *SXxx, data_t *SXxy, data_t *SXxz, data_t *SYxy, data_t *SYyy, data_t *SYyz, data_t *SZxz, data_t *SZyz, data_t *SZzz,
	data_t *e_dxi, data_t *dxi, data_t *e_dxi2, data_t *dxi2, data_t *e_dyj, data_t *dyj, data_t *e_dyj2, data_t *dyj2, data_t *e_dzk, data_t *dzk, data_t *dzk2, data_t *e_dzk2,
	data_t *ss, data_t *vwx, data_t *vwy, data_t *vwz, data_t *vwx2, data_t *vwy2, data_t *vwz2, data_t *SXss, data_t *SYss, data_t *SZss, data_t *pmlxss, data_t *pmlyss, data_t *pmlzss,
	data_t *VelocityWParameter1x, data_t *VelocityWParameter1y, data_t *VelocityWParameter1z, data_t *VelocityWParameter2x, data_t *VelocityWParameter2y, data_t *VelocityWParameter2z, data_t *VelocityWParameter3x, data_t *VelocityWParameter3y, data_t *VelocityWParameter3z,
	data_t *VelocityUParameter1x, data_t *VelocityUParameter1y, data_t *VelocityUParameter1z, data_t *VelocityUParameter2x, data_t *VelocityUParameter2y, data_t *VelocityUParameter2z, float DT, int nxt, int nyt, int nzt, int partIndex)
{
	float x1, x2, x3;
	float z1, z2, z3;
	float y1, y2, y3;
	float s1, s2, s3;
	float H = 100;
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = ix + iy * nxt + (iz + partIndex * nzt) * nxt * nyt;
	int offset_b = ix + iy * nxt + ((iz + partIndex * nzt) - 1) * nxt * nyt;//婵炴埊鎷??
	int offset_r = ix + 1 + iy * nxt + (iz + partIndex * nzt) * nxt * nyt;//闂侀潻鎷??
	int offset_h = ix + (iy - 1) * nxt + (iz + partIndex * nzt) * nxt * nyt;//闂侀潻鎷??
	int offset_q = ix + (iy + 1) * nxt + (iz + partIndex * nzt) * nxt * nyt;//闂侀潻鎷??
	int offset_l = ix - 1 + iy * nxt + (iz + partIndex * nzt) * nxt * nyt;//閻庣櫢鎷??
	int offset_u = ix + iy * nxt + (1 + (iz + partIndex * nzt)) * nxt * nyt;//婵炴埊鎷??

	if(ix > 0 && iy > 0 && iz >= 0 && ix < (nxt - 1) && iy < (nyt - 1) && iz <= (nzt - 1))
	{
		x1 = ((float)txx[offset_r] - (float)txx[offset]) * H;
		x2 = ((float)txy[offset] - (float)txy[offset_h]) * H;
		x3 = ((float)txz[offset] - (float)txz[offset_b]) * H;
		s1 = ((float)ss[offset_r] - (float)ss[offset]) * H;

		y1 = ((float)tyy[offset_q] - (float)tyy[offset]) * H;
		y2 = ((float)txy[offset] - (float)txy[offset_l]) * H;
		y3 = ((float)tyz[offset] - (float)tyz[offset_b]) * H;
		s2 = ((float)ss[offset_q] - (float)ss[offset]) * H;

		z1 = ((float)tzz[offset_u] - (float)tzz[offset]) * H;
		z2 = ((float)txz[offset] - (float)txz[offset_l]) * H;
		z3 = ((float)tyz[offset] - (float)tyz[offset_h]) * H;
		s3 = ((float)ss[offset_u] - (float)ss[offset]) * H;


		pmlxSxx[offset] = (float)pmlxSxx[offset] * (float)e_dxi2[offset] * d_Cpmll + (-DT * (float)dxi2[offset] * d_Cpmll * 0.5) * ((float)e_dxi2[offset] * d_Cpmll * (float)SXxx[offset] + x1);
		pmlySxy[offset] = (float)pmlySxy[offset] * (float)e_dyj[offset] * d_Cpmll + (-DT * (float)dyj[offset] * d_Cpmll * 0.5) * ((float)e_dyj[offset] * d_Cpmll * (float)SXxy[offset] + x2);
		pmlzSxz[offset] = (float)pmlzSxz[offset] * (float)e_dzk[offset] * d_Cpmll + (-DT * (float)dzk[offset] * d_Cpmll * 0.5) * ((float)e_dzk[offset] * d_Cpmll * (float)SXxz[offset] + x3);
		pmlxss[offset] = (float)pmlxss[offset] * (float)e_dxi2[offset] * d_Cpmll + (-DT * (float)dxi2[offset] * d_Cpmll * 0.5) * ((float)e_dxi2[offset] * d_Cpmll * (float)SXss[offset] + s1);
		SXxx[offset] = x1; SXxy[offset] = x2; SXxz[offset] = x3; SXss[offset] = s1;
		x1 = x1 + (float)pmlxSxx[offset];
		x2 = x2 + (float)pmlySxy[offset];
		x3 = x3 + (float)pmlzSxz[offset];
		s1 = s1 + (float)pmlxss[offset];

		pmlxSxy[offset] = (float)pmlxSxy[offset] * (float)e_dxi[offset] * d_Cpmll + (-DT * (float)dxi[offset] * d_Cpmll * 0.5) * ((float)e_dxi[offset]  * d_Cpmll* (float)SYxy[offset] + y2);
		pmlySyy[offset] = (float)pmlySyy[offset] * (float)e_dyj2[offset] * d_Cpmll + (-DT * (float)dyj2[offset]  * d_Cpmll* 0.5) * ((float)e_dyj2[offset]  * d_Cpmll* (float)SYyy[offset] + y1);
		pmlzSyz[offset] = (float)pmlzSyz[offset] * (float)e_dzk[offset] * d_Cpmll + (-DT * (float)dzk[offset] * d_Cpmll * 0.5) * ((float)e_dzk[offset] * d_Cpmll * (float)SYyz[offset] + y3);
		pmlyss[offset] = (float)pmlyss[offset] * (float)e_dyj2[offset] * d_Cpmll + (-DT * (float)dyj2[offset] * d_Cpmll * 0.5) * ((float)e_dyj2[offset]  * d_Cpmll* (float)SYss[offset] + s2);
		SYxy[offset] = y2; SYyy[offset] = y1; SYyz[offset] = y3; SYss[offset] = s2;
		y2 = y2 + (float)pmlxSxy[offset];
		y1 = y1 + (float)pmlySyy[offset];
		y3 = y3 + (float)pmlzSyz[offset];
		s2 = s2 + (float)pmlyss[offset];

		pmlxSxz[offset] = (float)pmlxSxz[offset] * (float)e_dxi[offset] * d_Cpmll + (-DT * (float)dxi[offset] * d_Cpmll * 0.5) * ((float)e_dxi[offset]  * d_Cpmll* (float)SZxz[offset] + z2);
		pmlySyz[offset] = (float)pmlySyz[offset] * (float)e_dyj[offset] * d_Cpmll + (-DT * (float)dyj[offset] * d_Cpmll * 0.5) * ((float)e_dyj[offset] * d_Cpmll * (float)SZyz[offset] + z3);
		pmlzSzz[offset] = (float)pmlzSzz[offset] * (float)e_dzk2[offset] * d_Cpmll + (-DT * (float)dzk2[offset] * d_Cpmll * 0.5) * ((float)e_dzk2[offset] * d_Cpmll * (float)SZzz[offset] + z1);
		pmlzss[offset] = (float)pmlzss[offset] * (float)e_dzk2[offset] * d_Cpmll + (-DT * (float)dzk2[offset]  * d_Cpmll* 0.5) * ((float)e_dzk2[offset]  * d_Cpmll* (float)SZss[offset] + s3);
		SZxz[offset] = z2; SZyz[offset] = z3; SZzz[offset] = z1; SZss[offset] = s3;
		z2 = z2 + (float)pmlxSxz[offset];
		z3 = z3 + (float)pmlySyz[offset];
		z1 = z1 + (float)pmlzSzz[offset];
		s3 = s3 + (float)pmlzss[offset];

		vwx[offset] = d_Cvwp1 * (float)VelocityWParameter1x[offset] * (float)vwx[offset] - d_Cvwp2 * (float)VelocityWParameter2x[offset] * (x1 + x2 + x3) - d_Cvwp3 * (float)VelocityWParameter3x[offset] * s1;
		vwy[offset] = d_Cvwp1 * (float)VelocityWParameter1y[offset] * (float)vwy[offset] - d_Cvwp2 * (float)VelocityWParameter2y[offset] * (y1 + y2 + y3) - d_Cvwp3 * (float)VelocityWParameter3y[offset] * s2;
		vwz[offset] = d_Cvwp1 * (float)VelocityWParameter1z[offset] * (float)vwz[offset] - d_Cvwp2 * (float)VelocityWParameter2z[offset] * (z1 + z2 + z3) - d_Cvwp3 * (float)VelocityWParameter3y[offset] * s3;


		vux[offset] = (float)vux[offset] + d_Cvup1 * (float)VelocityUParameter1x[offset] * (x1 + x2 + x3) - d_Cvup2 * (float)VelocityUParameter2x[offset] * ((float)vwx[offset] - (float)vwx2[offset]);
		vuy[offset] = (float)vuy[offset] + d_Cvup1 * (float)VelocityUParameter1y[offset] * (y1 + y2 + y3) - d_Cvup2 * (float)VelocityUParameter2y[offset] * ((float)vwy[offset] - (float)vwy2[offset]);
		vuz[offset] = (float)vuz[offset] + d_Cvup1 * (float)VelocityUParameter1z[offset] * (z1 + z2 + z3) - d_Cvup2 * (float)VelocityUParameter2z[offset] * ((float)vwz[offset] - (float)vwz2[offset]);
		vwx2[offset] = vwx[offset]; vwy2[offset] = vwy[offset]; vwz2[offset] = vwz[offset];

   }
}

__global__ void FD_Vbottom(data_t* vux, data_t* vuy, data_t* vuz,
	data_t* txx, data_t* tyy, data_t* tzz, data_t* txz, data_t* txy, data_t* tyz,
	data_t* pmlxSxx, data_t* pmlySxy, data_t* pmlzSxz, data_t* pmlxSxy, data_t* pmlySyy, data_t* pmlzSyz, data_t* pmlxSxz, data_t* pmlySyz, data_t* pmlzSzz,
	data_t* SXxx, data_t* SXxy, data_t* SXxz, data_t* SYxy, data_t* SYyy, data_t* SYyz, data_t* SZxz, data_t* SZyz, data_t* SZzz,
	data_t* e_dxi, data_t* dxi, data_t* e_dxi2, data_t* dxi2, data_t* e_dyj, data_t* dyj, data_t* e_dyj2, data_t* dyj2, data_t* e_dzk, data_t* dzk, data_t* dzk2, data_t* e_dzk2,
	data_t* ss, data_t* vwx, data_t* vwy, data_t* vwz, data_t* vwx2, data_t* vwy2, data_t* vwz2, data_t* SXss, data_t* SYss, data_t* SZss, data_t* pmlxss, data_t* pmlyss, data_t* pmlzss,
	data_t* VelocityWParameter1x, data_t* VelocityWParameter1y, data_t* VelocityWParameter1z, data_t* VelocityWParameter2x, data_t* VelocityWParameter2y, data_t* VelocityWParameter2z, data_t* VelocityWParameter3x, data_t* VelocityWParameter3y, data_t* VelocityWParameter3z,
	data_t* VelocityUParameter1x, data_t* VelocityUParameter1y, data_t* VelocityUParameter1z, data_t* VelocityUParameter2x, data_t* VelocityUParameter2y, data_t* VelocityUParameter2z, float DT, int nxt, int nyt, int nzt, int partIndex)
{
	float x1, x2, x3;
	float z1, z2, z3;
	float y1, y2, y3;
	float s1, s2, s3;
	float H = 100;
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = ix + iy * nxt + (iz + partIndex * nzt) * nxt * nyt;
	int offset_b = ix + iy * nxt + ((iz + partIndex * nzt) - 1) * nxt * nyt;//婵炴埊鎷??
	int offset_r = ix + 1 + iy * nxt + (iz + partIndex * nzt) * nxt * nyt;//闂侀潻鎷??
	int offset_h = ix + (iy - 1) * nxt + (iz + partIndex * nzt) * nxt * nyt;//闂侀潻鎷??
	int offset_q = ix + (iy + 1) * nxt + (iz + partIndex * nzt) * nxt * nyt;//闂侀潻鎷??
	int offset_l = ix - 1 + iy * nxt + (iz + partIndex * nzt) * nxt * nyt;//閻庣櫢鎷??
	int offset_u = ix + iy * nxt + (1 + (iz + partIndex * nzt)) * nxt * nyt;//婵炴埊鎷??

	if(ix > 0 && iy > 0 && iz > 0 && ix < (nxt - 1) && iy < (nyt - 1) && iz <= (nzt - 1))
	{
		x1 = ((float)txx[offset_r] - (float)txx[offset]) * H;
		x2 = ((float)txy[offset] - (float)txy[offset_h]) * H;
		x3 = ((float)txz[offset] - (float)txz[offset_b]) * H;
		s1 = ((float)ss[offset_r] - (float)ss[offset]) * H;

		y1 = ((float)tyy[offset_q] - (float)tyy[offset]) * H;
		y2 = ((float)txy[offset] - (float)txy[offset_l]) * H;
		y3 = ((float)tyz[offset] - (float)tyz[offset_b]) * H;
		s2 = ((float)ss[offset_q] - (float)ss[offset]) * H;

		z1 = ((float)tzz[offset_u] - (float)tzz[offset]) * H;
		z2 = ((float)txz[offset] - (float)txz[offset_l]) * H;
		z3 = ((float)tyz[offset] - (float)tyz[offset_h]) * H;
		s3 = ((float)ss[offset_u] - (float)ss[offset]) * H;


		pmlxSxx[offset] = (float)pmlxSxx[offset] * (float)e_dxi2[offset] * d_Cpmll + (-DT * (float)dxi2[offset] * d_Cpmll * 0.5) * ((float)e_dxi2[offset] * d_Cpmll * (float)SXxx[offset] + x1);
		pmlySxy[offset] = (float)pmlySxy[offset] * (float)e_dyj[offset] * d_Cpmll + (-DT * (float)dyj[offset] * d_Cpmll * 0.5) * ((float)e_dyj[offset] * d_Cpmll * (float)SXxy[offset] + x2);
		pmlzSxz[offset] = (float)pmlzSxz[offset] * (float)e_dzk[offset] * d_Cpmll + (-DT * (float)dzk[offset] * d_Cpmll * 0.5) * ((float)e_dzk[offset] * d_Cpmll * (float)SXxz[offset] + x3);
		pmlxss[offset] = (float)pmlxss[offset] * (float)e_dxi2[offset] * d_Cpmll + (-DT * (float)dxi2[offset] * d_Cpmll * 0.5) * ((float)e_dxi2[offset] * d_Cpmll * (float)SXss[offset] + s1);
		SXxx[offset] = x1; SXxy[offset] = x2; SXxz[offset] = x3; SXss[offset] = s1;
		x1 = x1 + (float)pmlxSxx[offset];
		x2 = x2 + (float)pmlySxy[offset];
		x3 = x3 + (float)pmlzSxz[offset];
		s1 = s1 + (float)pmlxss[offset];

		pmlxSxy[offset] = (float)pmlxSxy[offset] * (float)e_dxi[offset] * d_Cpmll + (-DT * (float)dxi[offset] * d_Cpmll * 0.5) * ((float)e_dxi[offset]  * d_Cpmll* (float)SYxy[offset] + y2);
		pmlySyy[offset] = (float)pmlySyy[offset] * (float)e_dyj2[offset] * d_Cpmll + (-DT * (float)dyj2[offset]  * d_Cpmll* 0.5) * ((float)e_dyj2[offset]  * d_Cpmll* (float)SYyy[offset] + y1);
		pmlzSyz[offset] = (float)pmlzSyz[offset] * (float)e_dzk[offset] * d_Cpmll + (-DT * (float)dzk[offset] * d_Cpmll * 0.5) * ((float)e_dzk[offset] * d_Cpmll * (float)SYyz[offset] + y3);
		pmlyss[offset] = (float)pmlyss[offset] * (float)e_dyj2[offset] * d_Cpmll + (-DT * (float)dyj2[offset] * d_Cpmll * 0.5) * ((float)e_dyj2[offset]  * d_Cpmll* (float)SYss[offset] + s2);
		SYxy[offset] = y2; SYyy[offset] = y1; SYyz[offset] = y3; SYss[offset] = s2;
		y2 = y2 + (float)pmlxSxy[offset];
		y1 = y1 + (float)pmlySyy[offset];
		y3 = y3 + (float)pmlzSyz[offset];
		s2 = s2 + (float)pmlyss[offset];

		pmlxSxz[offset] = (float)pmlxSxz[offset] * (float)e_dxi[offset] * d_Cpmll + (-DT * (float)dxi[offset] * d_Cpmll * 0.5) * ((float)e_dxi[offset]  * d_Cpmll* (float)SZxz[offset] + z2);
		pmlySyz[offset] = (float)pmlySyz[offset] * (float)e_dyj[offset] * d_Cpmll + (-DT * (float)dyj[offset] * d_Cpmll * 0.5) * ((float)e_dyj[offset] * d_Cpmll * (float)SZyz[offset] + z3);
		pmlzSzz[offset] = (float)pmlzSzz[offset] * (float)e_dzk2[offset] * d_Cpmll + (-DT * (float)dzk2[offset] * d_Cpmll * 0.5) * ((float)e_dzk2[offset] * d_Cpmll * (float)SZzz[offset] + z1);
		pmlzss[offset] = (float)pmlzss[offset] * (float)e_dzk2[offset] * d_Cpmll + (-DT * (float)dzk2[offset]  * d_Cpmll* 0.5) * ((float)e_dzk2[offset]  * d_Cpmll* (float)SZss[offset] + s3);
		SZxz[offset] = z2; SZyz[offset] = z3; SZzz[offset] = z1; SZss[offset] = s3;
		z2 = z2 + (float)pmlxSxz[offset];
		z3 = z3 + (float)pmlySyz[offset];
		z1 = z1 + (float)pmlzSzz[offset];
		s3 = s3 + (float)pmlzss[offset];

		vwx[offset] = d_Cvwp1 * (float)VelocityWParameter1x[offset] * (float)vwx[offset] - d_Cvwp2 * (float)VelocityWParameter2x[offset] * (x1 + x2 + x3) - d_Cvwp3 * (float)VelocityWParameter3x[offset] * s1;
		vwy[offset] = d_Cvwp1 * (float)VelocityWParameter1y[offset] * (float)vwy[offset] - d_Cvwp2 * (float)VelocityWParameter2y[offset] * (y1 + y2 + y3) - d_Cvwp3 * (float)VelocityWParameter3y[offset] * s2;
		vwz[offset] = d_Cvwp1 * (float)VelocityWParameter1z[offset] * (float)vwz[offset] - d_Cvwp2 * (float)VelocityWParameter2z[offset] * (z1 + z2 + z3) - d_Cvwp3 * (float)VelocityWParameter3y[offset] * s3;


		vux[offset] = (float)vux[offset] + d_Cvup1 * (float)VelocityUParameter1x[offset] * (x1 + x2 + x3) - d_Cvup2 * (float)VelocityUParameter2x[offset] * ((float)vwx[offset] - (float)vwx2[offset]);
		vuy[offset] = (float)vuy[offset] + d_Cvup1 * (float)VelocityUParameter1y[offset] * (y1 + y2 + y3) - d_Cvup2 * (float)VelocityUParameter2y[offset] * ((float)vwy[offset] - (float)vwy2[offset]);
		vuz[offset] = (float)vuz[offset] + d_Cvup1 * (float)VelocityUParameter1z[offset] * (z1 + z2 + z3) - d_Cvup2 * (float)VelocityUParameter2z[offset] * ((float)vwz[offset] - (float)vwz2[offset]);
		vwx2[offset] = vwx[offset]; vwy2[offset] = vwy[offset]; vwz2[offset] = vwz[offset];

   }
}

//------------------------------------------------------------------------------------------------------------------
//computer stress
__global__ void FD_Ttop(data_t* vux, data_t* vuy, data_t* vuz, data_t* txx, data_t* tzz, data_t* tyy, data_t* txz, data_t* txy, data_t* tyz,
	data_t* pmlxVux, data_t* pmlyVuy, data_t* pmlzVuz, data_t* pmlxVuy, data_t* pmlyVux, data_t* pmlyVuz, data_t* pmlzVuy, data_t* pmlzVux, data_t* pmlxVuz,
	data_t* Vuxx, data_t* Vuxy, data_t* Vuxz, data_t* Vuyx, data_t* Vuyy, data_t* Vuyz, data_t* Vuzx, data_t* Vuzy, data_t* Vuzz,
	data_t* e_dxi, data_t* dxi, data_t* e_dxi2, data_t* dxi2, data_t* e_dyj2, data_t* dyj2, data_t* e_dyj, data_t* dyj, data_t* dzk2, data_t* e_dzk2, data_t* e_dzk, data_t* dzk,
	data_t* Vwxx, data_t* Vwyy, data_t* Vwzz, data_t* vwx, data_t* vwy, data_t* vwz, data_t* ss, data_t* pmlxVwx, data_t* pmlyVwy, data_t* pmlzVwz,
	data_t* PressParameter1, data_t* PressParameter2, data_t* StressParameter1, data_t* StressParameter2, data_t* StressParameter3, data_t* StressParameterxy, data_t* StressParameterxz, data_t* StressParameteryz, float DT, int nxt, int nyt, int nzt, int partIndex)
{
	float uxx, uyy, uzz;
	float uxy, uxz, uyx, uyz, uzx, uzy;
	float wx, wy, wz;
	float H = 100;
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = ix + iy * nxt + (iz + partIndex * nzt) * nxt * nyt;
	int offset_b = ix + iy * nxt + ((iz + partIndex * nzt) - 1) * nxt * nyt;//婵炴埊鎷??
	int offset_r = ix + 1 + iy * nxt + (iz + partIndex * nzt) * nxt * nyt;//闂侀潻鎷??
	int offset_h = ix + (iy - 1) * nxt + (iz + partIndex * nzt) * nxt * nyt;//闂侀潻鎷??
	int offset_q = ix + (iy + 1) * nxt + (iz + partIndex * nzt) * nxt * nyt;//闂侀潻鎷??
	int offset_l = ix - 1 + iy * nxt + (iz + partIndex * nzt) * nxt * nyt;//閻庣櫢鎷??
	int offset_u = ix + iy * nxt + (1 + (iz + partIndex * nzt)) * nxt * nyt;//婵炴埊鎷??


	if(ix > 0 && iy > 0 && iz >= 0 && ix < (nxt - 1) && iy < (nyt - 1) && iz < (nzt - 1))
	{
		uxx = ((float)vux[offset] - (float)vux[offset_l]) * H;
		uyy = ((float)vuy[offset] - (float)vuy[offset_h]) * H;
		uzz = ((float)vuz[offset] - (float)vuz[offset_b]) * H;

		wx = ((float)vwx[offset] - (float)vwx[offset_l]) * H;
		wy = ((float)vwy[offset] - (float)vwy[offset_h]) * H;
		wz = ((float)vwz[offset] - (float)vwz[offset_b]) * H;

		uxy = ((float)vux[offset_q] - (float)vux[offset]) * H;
		uyx = ((float)vuy[offset_r] - (float)vuy[offset]) * H;

		uxz = ((float)vux[offset_u] - (float)vux[offset]) * H;
		uzx = ((float)vuz[offset_r] - (float)vuz[offset]) * H;

		uyz = ((float)vuy[offset_u] - (float)vuy[offset]) * H;
		uzy = ((float)vuz[offset_q] - (float)vuz[offset]) * H;



		pmlxVux[offset] = (float)pmlxVux[offset] * (float)e_dxi[offset] * d_Cpmll + (-DT * (float)dxi[offset] * d_Cpmll * 0.5) * ((float)e_dxi[offset] * d_Cpmll * (float)Vuxx[offset] + uxx);
		pmlyVuy[offset] = (float)pmlyVuy[offset] * (float)e_dyj[offset] * d_Cpmll + (-DT * (float)dyj[offset] * d_Cpmll * 0.5) * ((float)e_dyj[offset] * d_Cpmll * (float)Vuyy[offset] + uyy);
		pmlzVuz[offset] = (float)pmlzVuz[offset] * (float)e_dzk[offset] * d_Cpmll + (-DT * (float)dzk[offset] * d_Cpmll * 0.5) * ((float)e_dzk[offset] * d_Cpmll * (float)Vuzz[offset] + uzz);
		Vuxx[offset] = uxx; Vuyy[offset] = uyy; Vuzz[offset] = uzz;
		uxx = uxx + (float)pmlxVux[offset];
		uyy = uyy + (float)pmlyVuy[offset];
		uzz = uzz + (float)pmlzVuz[offset];


		pmlxVwx[offset] = (float)pmlxVwx[offset] * (float)e_dxi[offset] * d_Cpmll + (-DT * (float)dxi[offset] * d_Cpmll * 0.5) * ((float)e_dxi[offset] * d_Cpmll * (float)Vwxx[offset] + wx);
		pmlyVwy[offset] = (float)pmlyVwy[offset] * (float)e_dyj[offset] * d_Cpmll + (-DT * (float)dyj[offset] * d_Cpmll * 0.5) * ((float)e_dyj[offset] * d_Cpmll * (float)Vwyy[offset] + wy);
		pmlzVwz[offset] = (float)pmlzVwz[offset] * (float)e_dzk[offset] * d_Cpmll + (-DT * (float)dzk[offset] * d_Cpmll * 0.5) * ((float)e_dzk[offset] * d_Cpmll * (float)Vwzz[offset] + wz);
		Vwxx[offset] = wx; Vwyy[offset] = wy; Vwzz[offset] = wz;
		wx = wx + (float)pmlxVwx[offset];
		wy = wy + (float)pmlyVwy[offset];
		wz = wz + (float)pmlzVwz[offset];

		pmlxVuy[offset] = (float)pmlxVuy[offset] * (float)e_dyj2[offset] * d_Cpmll + (-DT * (float)dyj2[offset] * d_Cpmll * 0.5) * ((float)e_dyj2[offset] * d_Cpmll * (float)Vuxy[offset] + uxy);
		pmlyVux[offset] = (float)pmlyVux[offset] * (float)e_dxi2[offset] * d_Cpmll + (-DT * (float)dxi2[offset] * d_Cpmll * 0.5) * ((float)e_dxi2[offset] * d_Cpmll * (float)Vuyx[offset] + uyx);
		Vuxy[offset] = uxy; Vuyx[offset] = uyx;
		uxy = uxy + (float)pmlxVuy[offset];
		uyx = uyx + (float)pmlyVux[offset];

		pmlxVuz[offset] = (float)pmlxVuz[offset] * (float)e_dzk2[offset] * d_Cpmll + (-DT * (float)dzk2[offset] * d_Cpmll * 0.5) * ((float)e_dzk2[offset] * d_Cpmll * (float)Vuxz[offset] + uxz);
		pmlzVux[offset] = (float)pmlzVux[offset] * (float)e_dxi2[offset] * d_Cpmll + (-DT * (float)dxi2[offset] * d_Cpmll * 0.5) * ((float)e_dxi2[offset] * d_Cpmll * (float)Vuzx[offset] + uzx);
		Vuxz[offset] = uxz; Vuzx[offset] = uzx;
		uxz = uxz + (float)pmlxVuz[offset];
		uzx = uzx + (float)pmlzVux[offset];

		pmlyVuz[offset] = (float)pmlyVuz[offset] * (float)e_dzk2[offset] * d_Cpmll + (-DT * (float)dzk2[offset] * d_Cpmll * 0.5) * ((float)e_dzk2[offset] * d_Cpmll * (float)Vuyz[offset] + uyz);
		pmlzVuy[offset] = (float)pmlzVuy[offset] * (float)e_dyj2[offset] * d_Cpmll + (-DT * (float)dyj2[offset] * d_Cpmll * 0.5) * ((float)e_dyj2[offset] * d_Cpmll * (float)Vuzy[offset] + uzy);
		Vuzy[offset] = uzy; Vuyz[offset] = uyz;
		uyz = uyz + (float)pmlyVuz[offset];
		uzy = uzy + (float)pmlzVuy[offset];

		ss[offset] = (float)ss[offset] - d_Cpp1 * (float)PressParameter1[offset] * (uxx + uyy + uzz) - d_Cpp2 * (float)PressParameter2[offset] * (wx + wy + wz);
		txx[offset] = (float)txx[offset] + d_Csp1 * (float)StressParameter1[offset] * (uyy + uzz) + d_Csp2 * (float)StressParameter2[offset] * uxx + d_Csp3 * (float)StressParameter3[offset] * (wx + wy + wz);
		tyy[offset] = (float)tyy[offset] + d_Csp1 * (float)StressParameter1[offset] * (uxx + uzz) + d_Csp2 * (float)StressParameter2[offset] * uyy + d_Csp3 * (float)StressParameter3[offset] * (wx + wy + wz);
		tzz[offset] = (float)tzz[offset] + d_Csp1 * (float)StressParameter1[offset] * (uxx + uyy) + d_Csp2 * (float)StressParameter2[offset] * uzz + d_Csp3 * (float)StressParameter3[offset] * (wx + wy + wz);
		txy[offset] = (float)txy[offset] + d_Csp4 * (float)StressParameterxy[offset] * (uxy + uyx);
		tyz[offset] = (float)tyz[offset] + d_Csp4 * (float)StressParameteryz[offset] * (uyz + uzy);
		txz[offset] = (float)txz[offset] + d_Csp4 * (float)StressParameterxz[offset] * (uxz + uzx);

	}
}

__global__ void FD_Tmid(data_t* vux, data_t* vuy, data_t* vuz, data_t* txx, data_t* tzz, data_t* tyy, data_t* txz, data_t* txy, data_t* tyz,
	data_t* pmlxVux, data_t* pmlyVuy, data_t* pmlzVuz, data_t* pmlxVuy, data_t* pmlyVux, data_t* pmlyVuz, data_t* pmlzVuy, data_t* pmlzVux, data_t* pmlxVuz,
	data_t* Vuxx, data_t* Vuxy, data_t* Vuxz, data_t* Vuyx, data_t* Vuyy, data_t* Vuyz, data_t* Vuzx, data_t* Vuzy, data_t* Vuzz,
	data_t* e_dxi, data_t* dxi, data_t* e_dxi2, data_t* dxi2, data_t* e_dyj2, data_t* dyj2, data_t* e_dyj, data_t* dyj, data_t* dzk2, data_t* e_dzk2, data_t* e_dzk, data_t* dzk,
	data_t* Vwxx, data_t* Vwyy, data_t* Vwzz, data_t* vwx, data_t* vwy, data_t* vwz, data_t* ss, data_t* pmlxVwx, data_t* pmlyVwy, data_t* pmlzVwz,
	data_t* PressParameter1, data_t* PressParameter2, data_t* StressParameter1, data_t* StressParameter2, data_t* StressParameter3, data_t* StressParameterxy, data_t* StressParameterxz, data_t* StressParameteryz, float DT, int nxt, int nyt, int nzt, int partIndex)
{
	float uxx, uyy, uzz;
	float uxy, uxz, uyx, uyz, uzx, uzy;
	float wx, wy, wz;
	float H = 100;
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = ix + iy * nxt + (iz + partIndex * nzt) * nxt * nyt;
	int offset_b = ix + iy * nxt + ((iz + partIndex * nzt) - 1) * nxt * nyt;//婵炴埊鎷??
	int offset_r = ix + 1 + iy * nxt + (iz + partIndex * nzt) * nxt * nyt;//闂侀潻鎷??
	int offset_h = ix + (iy - 1) * nxt + (iz + partIndex * nzt) * nxt * nyt;//闂侀潻鎷??
	int offset_q = ix + (iy + 1) * nxt + (iz + partIndex * nzt) * nxt * nyt;//闂侀潻鎷??
	int offset_l = ix - 1 + iy * nxt + (iz + partIndex * nzt) * nxt * nyt;//閻庣櫢鎷??
	int offset_u = ix + iy * nxt + (1 + (iz + partIndex * nzt)) * nxt * nyt;//婵炴埊鎷??


	if(ix > 0 && iy > 0 && iz >= 0 && ix < (nxt - 1) && iy < (nyt - 1) && iz <= (nzt - 1))
	{
		uxx = ((float)vux[offset] - (float)vux[offset_l]) * H;
		uyy = ((float)vuy[offset] - (float)vuy[offset_h]) * H;
		uzz = ((float)vuz[offset] - (float)vuz[offset_b]) * H;

		wx = ((float)vwx[offset] - (float)vwx[offset_l]) * H;
		wy = ((float)vwy[offset] - (float)vwy[offset_h]) * H;
		wz = ((float)vwz[offset] - (float)vwz[offset_b]) * H;

		uxy = ((float)vux[offset_q] - (float)vux[offset]) * H;
		uyx = ((float)vuy[offset_r] - (float)vuy[offset]) * H;

		uxz = ((float)vux[offset_u] - (float)vux[offset]) * H;
		uzx = ((float)vuz[offset_r] - (float)vuz[offset]) * H;

		uyz = ((float)vuy[offset_u] - (float)vuy[offset]) * H;
		uzy = ((float)vuz[offset_q] - (float)vuz[offset]) * H;



		pmlxVux[offset] = (float)pmlxVux[offset] * (float)e_dxi[offset] * d_Cpmll + (-DT * (float)dxi[offset] * d_Cpmll * 0.5) * ((float)e_dxi[offset] * d_Cpmll * (float)Vuxx[offset] + uxx);
		pmlyVuy[offset] = (float)pmlyVuy[offset] * (float)e_dyj[offset] * d_Cpmll + (-DT * (float)dyj[offset] * d_Cpmll * 0.5) * ((float)e_dyj[offset] * d_Cpmll * (float)Vuyy[offset] + uyy);
		pmlzVuz[offset] = (float)pmlzVuz[offset] * (float)e_dzk[offset] * d_Cpmll + (-DT * (float)dzk[offset] * d_Cpmll * 0.5) * ((float)e_dzk[offset] * d_Cpmll * (float)Vuzz[offset] + uzz);
		Vuxx[offset] = uxx; Vuyy[offset] = uyy; Vuzz[offset] = uzz;
		uxx = uxx + (float)pmlxVux[offset];
		uyy = uyy + (float)pmlyVuy[offset];
		uzz = uzz + (float)pmlzVuz[offset];


		pmlxVwx[offset] = (float)pmlxVwx[offset] * (float)e_dxi[offset] * d_Cpmll + (-DT * (float)dxi[offset] * d_Cpmll * 0.5) * ((float)e_dxi[offset] * d_Cpmll * (float)Vwxx[offset] + wx);
		pmlyVwy[offset] = (float)pmlyVwy[offset] * (float)e_dyj[offset] * d_Cpmll + (-DT * (float)dyj[offset] * d_Cpmll * 0.5) * ((float)e_dyj[offset] * d_Cpmll * (float)Vwyy[offset] + wy);
		pmlzVwz[offset] = (float)pmlzVwz[offset] * (float)e_dzk[offset] * d_Cpmll + (-DT * (float)dzk[offset] * d_Cpmll * 0.5) * ((float)e_dzk[offset] * d_Cpmll * (float)Vwzz[offset] + wz);
		Vwxx[offset] = wx; Vwyy[offset] = wy; Vwzz[offset] = wz;
		wx = wx + (float)pmlxVwx[offset];
		wy = wy + (float)pmlyVwy[offset];
		wz = wz + (float)pmlzVwz[offset];

		pmlxVuy[offset] = (float)pmlxVuy[offset] * (float)e_dyj2[offset] * d_Cpmll + (-DT * (float)dyj2[offset] * d_Cpmll * 0.5) * ((float)e_dyj2[offset] * d_Cpmll * (float)Vuxy[offset] + uxy);
		pmlyVux[offset] = (float)pmlyVux[offset] * (float)e_dxi2[offset] * d_Cpmll + (-DT * (float)dxi2[offset] * d_Cpmll * 0.5) * ((float)e_dxi2[offset] * d_Cpmll * (float)Vuyx[offset] + uyx);
		Vuxy[offset] = uxy; Vuyx[offset] = uyx;
		uxy = uxy + (float)pmlxVuy[offset];
		uyx = uyx + (float)pmlyVux[offset];

		pmlxVuz[offset] = (float)pmlxVuz[offset] * (float)e_dzk2[offset] * d_Cpmll + (-DT * (float)dzk2[offset] * d_Cpmll * 0.5) * ((float)e_dzk2[offset] * d_Cpmll * (float)Vuxz[offset] + uxz);
		pmlzVux[offset] = (float)pmlzVux[offset] * (float)e_dxi2[offset] * d_Cpmll + (-DT * (float)dxi2[offset] * d_Cpmll * 0.5) * ((float)e_dxi2[offset] * d_Cpmll * (float)Vuzx[offset] + uzx);
		Vuxz[offset] = uxz; Vuzx[offset] = uzx;
		uxz = uxz + (float)pmlxVuz[offset];
		uzx = uzx + (float)pmlzVux[offset];

		pmlyVuz[offset] = (float)pmlyVuz[offset] * (float)e_dzk2[offset] * d_Cpmll + (-DT * (float)dzk2[offset] * d_Cpmll * 0.5) * ((float)e_dzk2[offset] * d_Cpmll * (float)Vuyz[offset] + uyz);
		pmlzVuy[offset] = (float)pmlzVuy[offset] * (float)e_dyj2[offset] * d_Cpmll + (-DT * (float)dyj2[offset] * d_Cpmll * 0.5) * ((float)e_dyj2[offset] * d_Cpmll * (float)Vuzy[offset] + uzy);
		Vuzy[offset] = uzy; Vuyz[offset] = uyz;
		uyz = uyz + (float)pmlyVuz[offset];
		uzy = uzy + (float)pmlzVuy[offset];

		ss[offset] = (float)ss[offset] - d_Cpp1 * (float)PressParameter1[offset] * (uxx + uyy + uzz) - d_Cpp2 * (float)PressParameter2[offset] * (wx + wy + wz);
		txx[offset] = (float)txx[offset] + d_Csp1 * (float)StressParameter1[offset] * (uyy + uzz) + d_Csp2 * (float)StressParameter2[offset] * uxx + d_Csp3 * (float)StressParameter3[offset] * (wx + wy + wz);
		tyy[offset] = (float)tyy[offset] + d_Csp1 * (float)StressParameter1[offset] * (uxx + uzz) + d_Csp2 * (float)StressParameter2[offset] * uyy + d_Csp3 * (float)StressParameter3[offset] * (wx + wy + wz);
		tzz[offset] = (float)tzz[offset] + d_Csp1 * (float)StressParameter1[offset] * (uxx + uyy) + d_Csp2 * (float)StressParameter2[offset] * uzz + d_Csp3 * (float)StressParameter3[offset] * (wx + wy + wz);
		txy[offset] = (float)txy[offset] + d_Csp4 * (float)StressParameterxy[offset] * (uxy + uyx);
		tyz[offset] = (float)tyz[offset] + d_Csp4 * (float)StressParameteryz[offset] * (uyz + uzy);
		txz[offset] = (float)txz[offset] + d_Csp4 * (float)StressParameterxz[offset] * (uxz + uzx);

	}
}

__global__ void FD_Tbottom(data_t* vux, data_t* vuy, data_t* vuz, data_t* txx, data_t* tzz, data_t* tyy, data_t* txz, data_t* txy, data_t* tyz,
	data_t* pmlxVux, data_t* pmlyVuy, data_t* pmlzVuz, data_t* pmlxVuy, data_t* pmlyVux, data_t* pmlyVuz, data_t* pmlzVuy, data_t* pmlzVux, data_t* pmlxVuz,
	data_t* Vuxx, data_t* Vuxy, data_t* Vuxz, data_t* Vuyx, data_t* Vuyy, data_t* Vuyz, data_t* Vuzx, data_t* Vuzy, data_t* Vuzz,
	data_t* e_dxi, data_t* dxi, data_t* e_dxi2, data_t* dxi2, data_t* e_dyj2, data_t* dyj2, data_t* e_dyj, data_t* dyj, data_t* dzk2, data_t* e_dzk2, data_t* e_dzk, data_t* dzk,
	data_t* Vwxx, data_t* Vwyy, data_t* Vwzz, data_t* vwx, data_t* vwy, data_t* vwz, data_t* ss, data_t* pmlxVwx, data_t* pmlyVwy, data_t* pmlzVwz,
	data_t* PressParameter1, data_t* PressParameter2, data_t* StressParameter1, data_t* StressParameter2, data_t* StressParameter3, data_t* StressParameterxy, data_t* StressParameterxz, data_t* StressParameteryz, float DT, int nxt, int nyt, int nzt, int partIndex)
{
	float uxx, uyy, uzz;
	float uxy, uxz, uyx, uyz, uzx, uzy;
	float wx, wy, wz;
	float H = 100;
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = ix + iy * nxt + (iz + partIndex * nzt) * nxt * nyt;
	int offset_b = ix + iy * nxt + ((iz + partIndex * nzt) - 1) * nxt * nyt;//婵炴埊鎷??
	int offset_r = ix + 1 + iy * nxt + (iz + partIndex * nzt) * nxt * nyt;//闂侀潻鎷??
	int offset_h = ix + (iy - 1) * nxt + (iz + partIndex * nzt) * nxt * nyt;//闂侀潻鎷??
	int offset_q = ix + (iy + 1) * nxt + (iz + partIndex * nzt) * nxt * nyt;//闂侀潻鎷??
	int offset_l = ix - 1 + iy * nxt + (iz + partIndex * nzt) * nxt * nyt;//閻庣櫢鎷??
	int offset_u = ix + iy * nxt + (1 + (iz + partIndex * nzt)) * nxt * nyt;//婵炴埊鎷??


	if(ix > 0 && iy > 0 && iz > 0 && ix < (nxt - 1) && iy < (nyt - 1) && iz <= (nzt - 1))
	{
		uxx = ((float)vux[offset] - (float)vux[offset_l]) * H;
		uyy = ((float)vuy[offset] - (float)vuy[offset_h]) * H;
		uzz = ((float)vuz[offset] - (float)vuz[offset_b]) * H;

		wx = ((float)vwx[offset] - (float)vwx[offset_l]) * H;
		wy = ((float)vwy[offset] - (float)vwy[offset_h]) * H;
		wz = ((float)vwz[offset] - (float)vwz[offset_b]) * H;

		uxy = ((float)vux[offset_q] - (float)vux[offset]) * H;
		uyx = ((float)vuy[offset_r] - (float)vuy[offset]) * H;

		uxz = ((float)vux[offset_u] - (float)vux[offset]) * H;
		uzx = ((float)vuz[offset_r] - (float)vuz[offset]) * H;

		uyz = ((float)vuy[offset_u] - (float)vuy[offset]) * H;
		uzy = ((float)vuz[offset_q] - (float)vuz[offset]) * H;



		pmlxVux[offset] = (float)pmlxVux[offset] * (float)e_dxi[offset] * d_Cpmll + (-DT * (float)dxi[offset] * d_Cpmll * 0.5) * ((float)e_dxi[offset] * d_Cpmll * (float)Vuxx[offset] + uxx);
		pmlyVuy[offset] = (float)pmlyVuy[offset] * (float)e_dyj[offset] * d_Cpmll + (-DT * (float)dyj[offset] * d_Cpmll * 0.5) * ((float)e_dyj[offset] * d_Cpmll * (float)Vuyy[offset] + uyy);
		pmlzVuz[offset] = (float)pmlzVuz[offset] * (float)e_dzk[offset] * d_Cpmll + (-DT * (float)dzk[offset] * d_Cpmll * 0.5) * ((float)e_dzk[offset] * d_Cpmll * (float)Vuzz[offset] + uzz);
		Vuxx[offset] = uxx; Vuyy[offset] = uyy; Vuzz[offset] = uzz;
		uxx = uxx + (float)pmlxVux[offset];
		uyy = uyy + (float)pmlyVuy[offset];
		uzz = uzz + (float)pmlzVuz[offset];


		pmlxVwx[offset] = (float)pmlxVwx[offset] * (float)e_dxi[offset] * d_Cpmll + (-DT * (float)dxi[offset] * d_Cpmll * 0.5) * ((float)e_dxi[offset] * d_Cpmll * (float)Vwxx[offset] + wx);
		pmlyVwy[offset] = (float)pmlyVwy[offset] * (float)e_dyj[offset] * d_Cpmll + (-DT * (float)dyj[offset] * d_Cpmll * 0.5) * ((float)e_dyj[offset] * d_Cpmll * (float)Vwyy[offset] + wy);
		pmlzVwz[offset] = (float)pmlzVwz[offset] * (float)e_dzk[offset] * d_Cpmll + (-DT * (float)dzk[offset] * d_Cpmll * 0.5) * ((float)e_dzk[offset] * d_Cpmll * (float)Vwzz[offset] + wz);
		Vwxx[offset] = wx; Vwyy[offset] = wy; Vwzz[offset] = wz;
		wx = wx + (float)pmlxVwx[offset];
		wy = wy + (float)pmlyVwy[offset];
		wz = wz + (float)pmlzVwz[offset];

		pmlxVuy[offset] = (float)pmlxVuy[offset] * (float)e_dyj2[offset] * d_Cpmll + (-DT * (float)dyj2[offset] * d_Cpmll * 0.5) * ((float)e_dyj2[offset] * d_Cpmll * (float)Vuxy[offset] + uxy);
		pmlyVux[offset] = (float)pmlyVux[offset] * (float)e_dxi2[offset] * d_Cpmll + (-DT * (float)dxi2[offset] * d_Cpmll * 0.5) * ((float)e_dxi2[offset] * d_Cpmll * (float)Vuyx[offset] + uyx);
		Vuxy[offset] = uxy; Vuyx[offset] = uyx;
		uxy = uxy + (float)pmlxVuy[offset];
		uyx = uyx + (float)pmlyVux[offset];

		pmlxVuz[offset] = (float)pmlxVuz[offset] * (float)e_dzk2[offset] * d_Cpmll + (-DT * (float)dzk2[offset] * d_Cpmll * 0.5) * ((float)e_dzk2[offset] * d_Cpmll * (float)Vuxz[offset] + uxz);
		pmlzVux[offset] = (float)pmlzVux[offset] * (float)e_dxi2[offset] * d_Cpmll + (-DT * (float)dxi2[offset] * d_Cpmll * 0.5) * ((float)e_dxi2[offset] * d_Cpmll * (float)Vuzx[offset] + uzx);
		Vuxz[offset] = uxz; Vuzx[offset] = uzx;
		uxz = uxz + (float)pmlxVuz[offset];
		uzx = uzx + (float)pmlzVux[offset];

		pmlyVuz[offset] = (float)pmlyVuz[offset] * (float)e_dzk2[offset] * d_Cpmll + (-DT * (float)dzk2[offset] * d_Cpmll * 0.5) * ((float)e_dzk2[offset] * d_Cpmll * (float)Vuyz[offset] + uyz);
		pmlzVuy[offset] = (float)pmlzVuy[offset] * (float)e_dyj2[offset] * d_Cpmll + (-DT * (float)dyj2[offset] * d_Cpmll * 0.5) * ((float)e_dyj2[offset] * d_Cpmll * (float)Vuzy[offset] + uzy);
		Vuzy[offset] = uzy; Vuyz[offset] = uyz;
		uyz = uyz + (float)pmlyVuz[offset];
		uzy = uzy + (float)pmlzVuy[offset];

		ss[offset] = (float)ss[offset] - d_Cpp1 * (float)PressParameter1[offset] * (uxx + uyy + uzz) - d_Cpp2 * (float)PressParameter2[offset] * (wx + wy + wz);
		txx[offset] = (float)txx[offset] + d_Csp1 * (float)StressParameter1[offset] * (uyy + uzz) + d_Csp2 * (float)StressParameter2[offset] * uxx + d_Csp3 * (float)StressParameter3[offset] * (wx + wy + wz);
		tyy[offset] = (float)tyy[offset] + d_Csp1 * (float)StressParameter1[offset] * (uxx + uzz) + d_Csp2 * (float)StressParameter2[offset] * uyy + d_Csp3 * (float)StressParameter3[offset] * (wx + wy + wz);
		tzz[offset] = (float)tzz[offset] + d_Csp1 * (float)StressParameter1[offset] * (uxx + uyy) + d_Csp2 * (float)StressParameter2[offset] * uzz + d_Csp3 * (float)StressParameter3[offset] * (wx + wy + wz);
		txy[offset] = (float)txy[offset] + d_Csp4 * (float)StressParameterxy[offset] * (uxy + uyx);
		tyz[offset] = (float)tyz[offset] + d_Csp4 * (float)StressParameteryz[offset] * (uyz + uzy);
		txz[offset] = (float)txz[offset] + d_Csp4 * (float)StressParameterxz[offset] * (uxz + uzx);

	}
}
//-------------------------------------------------------------------------------------------------------------------------------
//send and recive halo 
void Send_halo_top_V(int myid, int *coord, blockInfo *block_info, MPI_Request *req_send, MPI_Request *req_recv, int *send_count, int *recv_count, int up, int down, int back, int forward, int left, int right, MPI_Comm MCW, int nxt, int nyt, int nzt){

	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;
	if(coord[2] != Zmax - 1){
		MPI_Isend(block_info->halo_Vuz_up_pack, (nxt * nyt) * 2, MPI_BYTE, up, 7, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Vwz_up_pack, (nxt * nyt) * 2, MPI_BYTE, up, 23, MCW, &req_send[(*send_count)++]);
	}
	
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榶閺夌偟顥愮粩鐔虹磽濮掝槒n闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != 0){

		MPI_Isend(block_info->halo_Vux_back_top_pack, (nxt * nzt) * 2, MPI_BYTE, back, 6, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Vuz_back_top_pack, (nxt * nzt) * 2, MPI_BYTE, back, 10, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ヨ姤�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != Ymax - 1){
		MPI_Isend(block_info->halo_Vuy_front_top_pack, (nxt * nzt) * 2, MPI_BYTE, forward, 9, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Vwy_front_top_pack, (nxt * nzt) * 2, MPI_BYTE, forward, 22, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖in闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != 0){
		MPI_Isend(block_info->halo_Vuy_left_top_pack, (nyt * nzt) * 2, MPI_BYTE, left, 2, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Vuz_left_top_pack, (nyt * nzt) * 2, MPI_BYTE, left, 4, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != Xmax - 1){
		MPI_Isend(block_info->halo_Vux_right_top_pack, (nyt * nzt) * 2, MPI_BYTE, right, 1, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Vwx_right_top_pack, (nyt * nzt) * 2, MPI_BYTE, right, 21, MCW, &req_send[(*send_count)++]);
	}
}

void Recv_halo_top_V(int myid, int *coord, blockInfo *block_info, MPI_Request *req_send, MPI_Request *req_recv, int *send_count, int *recv_count, int up, int down, int back, int forward, int left, int right, MPI_Comm MCW, int nxt, int nyt, int nzt){

	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;
	if(coord[2] != Zmax - 1){
		MPI_Irecv(block_info->halo_Vux_down_recv, (nxt * nyt) * 2, MPI_BYTE, up, 3, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Vuy_down_recv, (nxt * nyt) * 2, MPI_BYTE, up, 5, MCW, &req_recv[(*recv_count)++]);
	}
	
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榶閺夌偟顥愮粩鐔虹磽濮掝槒n闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != 0){
		MPI_Irecv(block_info->halo_Vuy_front_top_recv, (nxt * nzt) * 2, MPI_BYTE, back, 9, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Vwy_front_top_recv, (nxt * nzt) * 2, MPI_BYTE, back, 22, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ヨ姤�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != Ymax - 1){
		MPI_Irecv(block_info->halo_Vux_back_top_recv, (nxt * nzt) * 2, MPI_BYTE, forward, 6, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Vuz_back_top_recv, (nxt * nzt) * 2, MPI_BYTE, forward, 10, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖in闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != 0){
		MPI_Irecv(block_info->halo_Vux_right_top_recv, (nyt * nzt) * 2, MPI_BYTE, left, 1, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Vwx_right_top_recv, (nyt * nzt) * 2, MPI_BYTE, left, 21, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != Xmax - 1){
		MPI_Irecv(block_info->halo_Vuy_left_top_recv, (nyt * nzt) * 2, MPI_BYTE, right, 2, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Vuz_left_top_recv, (nyt * nzt) * 2, MPI_BYTE, right, 4, MCW, &req_recv[(*recv_count)++]);
	}
}

void Send_halo_top_T(int myid, int *coord, blockInfo *block_info, MPI_Request *req_send, MPI_Request *req_recv, int *send_count, int *recv_count, int up, int down, int back, int forward, int left, int right, MPI_Comm MCW, int nxt, int nyt, int nzt){

	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;
	if(coord[2] != Zmax - 1){
		MPI_Isend(block_info->halo_Txz_up_pack, (nxt * nyt) * 2, MPI_BYTE, up, 19, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Tyz_up_pack, (nxt * nyt) * 2, MPI_BYTE, up, 20, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榶閺夌偟顥愮粩鐔虹磽濮掝槒n闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != 0){

		MPI_Isend(block_info->halo_Tyy_back_top_pack, (nxt * nzt) * 2, MPI_BYTE, back, 15, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_ss_back_top_pack, (nxt * nzt) * 2, MPI_BYTE, back, 25, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ヨ姤�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != Ymax - 1){
		MPI_Isend(block_info->halo_Tyz_front_top_pack, (nxt * nzt) * 2, MPI_BYTE, forward, 16, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Txy_front_top_pack, (nxt * nzt) * 2, MPI_BYTE, forward, 17, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖in闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != 0){
		MPI_Isend(block_info->halo_Txx_left_top_pack, (nyt * nzt) * 2, MPI_BYTE, left, 14, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_ss_left_top_pack, (nyt * nzt) * 2, MPI_BYTE, left, 24, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != Xmax - 1){
		MPI_Isend(block_info->halo_Txy_right_top_pack, (nyt * nzt) * 2, MPI_BYTE, right, 12, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Txz_right_top_pack, (nyt * nzt) * 2, MPI_BYTE, right, 13, MCW, &req_send[(*send_count)++]);
	}
}

void Recv_halo_top_T(int myid, int *coord, blockInfo *block_info, MPI_Request *req_send, MPI_Request *req_recv, int *send_count, int *recv_count, int up, int down, int back, int forward, int left, int right, MPI_Comm MCW, int nxt, int nyt, int nzt){

	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;
	if(coord[2] != Zmax - 1){
		MPI_Irecv(block_info->halo_Tzz_down_recv, (nxt * nyt) * 2, MPI_BYTE, up, 18, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_ss_down_recv, (nxt * nyt) * 2, MPI_BYTE, up, 26, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榶閺夌偟顥愮粩鐔虹磽濮掝槒n闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != 0){
		MPI_Irecv(block_info->halo_Tyz_front_top_recv, (nxt * nzt) * 2, MPI_BYTE, back, 16, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Txy_front_top_recv, (nxt * nzt) * 2, MPI_BYTE, back, 17, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ヨ姤�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != Ymax - 1){
		MPI_Irecv(block_info->halo_Tyy_back_top_recv, (nxt * nzt) * 2, MPI_BYTE, forward, 15, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_ss_back_top_recv, (nxt * nzt) * 2, MPI_BYTE, forward, 25, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖in闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != 0){
		MPI_Irecv(block_info->halo_Txy_right_top_recv, (nyt * nzt) * 2, MPI_BYTE, left, 12, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Txz_right_top_recv, (nyt * nzt) * 2, MPI_BYTE, left, 13, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != Xmax - 1){
		MPI_Irecv(block_info->halo_Txx_left_top_recv, (nyt * nzt) * 2, MPI_BYTE, right, 14, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_ss_left_top_recv, (nyt * nzt) * 2, MPI_BYTE, right, 24, MCW, &req_recv[(*recv_count)++]);
	}
}

void Send_halo_mid_V(int myid, int *coord, blockInfo *block_info, MPI_Request *req_send, MPI_Request *req_recv, int *send_count, int *recv_count, int up, int down, int back, int forward, int left, int right, MPI_Comm MCW, int nxt, int nyt, int nzt){
		
	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榶閺夌偟顥愮粩鐔虹磽濮掝槒n闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != 0){

		MPI_Isend(block_info->halo_Vux_back_mid_pack, (nxt * nzt) * 2, MPI_BYTE, back, 30, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Vuz_back_mid_pack, (nxt * nzt) * 2, MPI_BYTE, back, 31, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ヨ姤�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != Ymax - 1){
		MPI_Isend(block_info->halo_Vuy_front_mid_pack, (nxt * nzt) * 2, MPI_BYTE, forward, 32, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Vwy_front_mid_pack, (nxt * nzt) * 2, MPI_BYTE, forward, 33, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖in闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != 0){
		MPI_Isend(block_info->halo_Vuy_left_mid_pack, (nyt * nzt) * 2, MPI_BYTE, left, 34, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Vuz_left_mid_pack, (nyt * nzt) * 2, MPI_BYTE, left, 35, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != Xmax - 1){
		MPI_Isend(block_info->halo_Vux_right_mid_pack, (nyt * nzt) * 2, MPI_BYTE, right, 36, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Vwx_right_mid_pack, (nyt * nzt) * 2, MPI_BYTE, right, 37, MCW, &req_send[(*send_count)++]);
	}
}

void Recv_halo_mid_V(int myid, int *coord, blockInfo *block_info, MPI_Request *req_send, MPI_Request *req_recv, int *send_count, int *recv_count, int up, int down, int back, int forward, int left, int right, MPI_Comm MCW, int nxt, int nyt, int nzt){
		
	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榶閺夌偟顥愮粩鐔虹磽濮掝槒n闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != 0){
		MPI_Irecv(block_info->halo_Vuy_front_mid_recv, (nxt * nzt) * 2, MPI_BYTE, back, 32, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Vwy_front_mid_recv, (nxt * nzt) * 2, MPI_BYTE, back, 33, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ヨ姤�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != Ymax - 1){
		MPI_Irecv(block_info->halo_Vux_back_mid_recv, (nxt * nzt) * 2, MPI_BYTE, forward, 30, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Vuz_back_mid_recv, (nxt * nzt) * 2, MPI_BYTE, forward, 31, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖in闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != 0){
		MPI_Irecv(block_info->halo_Vux_right_mid_recv, (nyt * nzt) * 2, MPI_BYTE, left, 36, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Vwx_right_mid_recv, (nyt * nzt) * 2, MPI_BYTE, left, 37, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != Xmax - 1){
		MPI_Irecv(block_info->halo_Vuy_left_mid_recv, (nyt * nzt) * 2, MPI_BYTE, right, 34, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Vuz_left_mid_recv, (nyt * nzt) * 2, MPI_BYTE, right, 35, MCW, &req_recv[(*recv_count)++]);
	}
}

void Send_halo_mid_T(int myid, int *coord, blockInfo *block_info, MPI_Request *req_send, MPI_Request *req_recv, int *send_count, int *recv_count, int up, int down, int back, int forward, int left, int right, MPI_Comm MCW, int nxt, int nyt, int nzt){
		
	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榶閺夌偟顥愮粩鐔虹磽濮掝槒n闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != 0){

		MPI_Isend(block_info->halo_Tyy_back_mid_pack, (nxt * nzt) * 2, MPI_BYTE, back, 50, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_ss_back_mid_pack, (nxt * nzt) * 2, MPI_BYTE, back, 51, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ヨ姤�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != Ymax - 1){
		MPI_Isend(block_info->halo_Tyz_front_mid_pack, (nxt * nzt) * 2, MPI_BYTE, forward, 52, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Txy_front_mid_pack, (nxt * nzt) * 2, MPI_BYTE, forward, 53, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖in闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != 0){
		MPI_Isend(block_info->halo_Txx_left_mid_pack, (nyt * nzt) * 2, MPI_BYTE, left, 54, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_ss_left_mid_pack, (nyt * nzt) * 2, MPI_BYTE, left, 55, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != Xmax - 1){
		MPI_Isend(block_info->halo_Txy_right_mid_pack, (nyt * nzt) * 2, MPI_BYTE, right, 56, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Txz_right_mid_pack, (nyt * nzt) * 2, MPI_BYTE, right, 57, MCW, &req_send[(*send_count)++]);
	}
}

void Recv_halo_mid_T(int myid, int *coord, blockInfo *block_info, MPI_Request *req_send, MPI_Request *req_recv, int *send_count, int *recv_count, int up, int down, int back, int forward, int left, int right, MPI_Comm MCW, int nxt, int nyt, int nzt){
		
	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榶閺夌偟顥愮粩鐔虹磽濮掝槒n闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != 0){
		MPI_Irecv(block_info->halo_Tyz_front_mid_recv, (nxt * nzt) * 2, MPI_BYTE, back, 52, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Txy_front_mid_recv, (nxt * nzt) * 2, MPI_BYTE, back, 53, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ヨ姤�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != Ymax - 1){
		MPI_Irecv(block_info->halo_Tyy_back_mid_recv, (nxt * nzt) * 2, MPI_BYTE, forward, 50, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_ss_back_mid_recv, (nxt * nzt) * 2, MPI_BYTE, forward, 51, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖in闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != 0){
		MPI_Irecv(block_info->halo_Txy_right_mid_recv, (nyt * nzt) * 2, MPI_BYTE, left, 56, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Txz_right_mid_recv, (nyt * nzt) * 2, MPI_BYTE, left, 57, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != Xmax - 1){
		MPI_Irecv(block_info->halo_Txx_left_mid_recv, (nyt * nzt) * 2, MPI_BYTE, right, 54, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_ss_left_mid_recv, (nyt * nzt) * 2, MPI_BYTE, right, 55, MCW, &req_recv[(*recv_count)++]);
	}
}

void Send_halo_bottom_V(int myid, int *coord, blockInfo *block_info, MPI_Request *req_send, MPI_Request *req_recv, int *send_count, int *recv_count, int up, int down, int back, int forward, int left, int right, MPI_Comm MCW, int nxt, int nyt, int nzt){

	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;
	//闁兼眹鍎扮粭澶愬及閻ョ偓�???€涙ɑ浠樺☉鎾愁儔濞间即鎯冮崟顔剧?缂佸?�???
	if(coord[2] != 0){
		MPI_Isend(block_info->halo_Vux_down_pack, (nxt * nyt) * 2, MPI_BYTE, down, 3, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Vuy_down_pack, (nxt * nyt) * 2, MPI_BYTE, down, 5, MCW, &req_send[(*send_count)++]);
	}
		//闁兼眹鍎扮粭澶愬及椤栨稒浠榶閺夌偟顥愮粩鐔虹磽濮掝槒n闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != 0){

		MPI_Isend(block_info->halo_Vux_back_bottom_pack, (nxt * nzt) * 2, MPI_BYTE, back, 40, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Vuz_back_bottom_pack, (nxt * nzt) * 2, MPI_BYTE, back, 41, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ヨ姤�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != Ymax - 1){
		MPI_Isend(block_info->halo_Vuy_front_bottom_pack, (nxt * nzt) * 2, MPI_BYTE, forward, 42, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Vwy_front_bottom_pack, (nxt * nzt) * 2, MPI_BYTE, forward, 43, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖in闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != 0){
		MPI_Isend(block_info->halo_Vuy_left_bottom_pack, (nyt * nzt) * 2, MPI_BYTE, left, 44, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Vuz_left_bottom_pack, (nyt * nzt) * 2, MPI_BYTE, left, 45, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != Xmax - 1){
		MPI_Isend(block_info->halo_Vux_right_bottom_pack, (nyt * nzt) * 2, MPI_BYTE, right, 46, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Vwx_right_bottom_pack, (nyt * nzt) * 2, MPI_BYTE, right, 47, MCW, &req_send[(*send_count)++]);
	}
	
	
}

void Recv_halo_bottom_V(int myid, int *coord, blockInfo *block_info, MPI_Request *req_send, MPI_Request *req_recv, int *send_count, int *recv_count, int up, int down, int back, int forward, int left, int right, MPI_Comm MCW, int nxt, int nyt, int nzt){

	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;
	//闁兼眹鍎扮粭澶愬及閻ョ偓�???€涙ɑ浠樺☉鎾愁儔濞间即鎯冮崟顔剧?缂佸?�???
	if(coord[2] != 0){
		MPI_Irecv(block_info->halo_Vuz_up_recv, (nxt * nyt) * 2, MPI_BYTE, down, 7, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Vwz_up_recv, (nxt * nyt) * 2, MPI_BYTE, down, 23, MCW, &req_recv[(*recv_count)++]);
	}
		//闁兼眹鍎扮粭澶愬及椤栨稒浠榶閺夌偟顥愮粩鐔虹磽濮掝槒n闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != 0){
		MPI_Irecv(block_info->halo_Vuy_front_bottom_recv, (nxt * nzt) * 2, MPI_BYTE, back, 42, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Vwy_front_bottom_recv, (nxt * nzt) * 2, MPI_BYTE, back, 43, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ヨ姤�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != Ymax - 1){
		MPI_Irecv(block_info->halo_Vux_back_bottom_recv, (nxt * nzt) * 2, MPI_BYTE, forward, 40, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Vuz_back_bottom_recv, (nxt * nzt) * 2, MPI_BYTE, forward, 41, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖in闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != 0){
		MPI_Irecv(block_info->halo_Vux_right_bottom_recv, (nyt * nzt) * 2, MPI_BYTE, left, 46, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Vwx_right_bottom_recv, (nyt * nzt) * 2, MPI_BYTE, left, 47, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != Xmax - 1){
		MPI_Irecv(block_info->halo_Vuy_left_bottom_recv, (nyt * nzt) * 2, MPI_BYTE, right, 44, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Vuz_left_bottom_recv, (nyt * nzt) * 2, MPI_BYTE, right, 45, MCW, &req_recv[(*recv_count)++]);
	}
	
	
}

void Send_halo_bottom_T(int myid, int *coord, blockInfo *block_info, MPI_Request *req_send, MPI_Request *req_recv, int *send_count, int *recv_count, int up, int down, int back, int forward, int left, int right, MPI_Comm MCW, int nxt, int nyt, int nzt){

	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;
	//闁兼眹鍎扮粭澶愬及閻ョ偓�???€涙ɑ浠樺☉鎾愁儔濞间即鎯冮崟顔剧?缂佸?�???
	if(coord[2] != 0){
		MPI_Isend(block_info->halo_Tzz_down_pack, (nxt * nyt) * 2, MPI_BYTE, down, 18, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_ss_down_pack, (nxt * nyt) * 2, MPI_BYTE, down, 26, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榶閺夌偟顥愮粩鐔虹磽濮掝槒n闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != 0){

		MPI_Isend(block_info->halo_Tyy_back_bottom_pack, (nxt * nzt) * 2, MPI_BYTE, back, 60, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_ss_back_bottom_pack, (nxt * nzt) * 2, MPI_BYTE, back, 61, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ヨ姤�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != Ymax - 1){
		MPI_Isend(block_info->halo_Tyz_front_bottom_pack, (nxt * nzt) * 2, MPI_BYTE, forward, 62, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Txy_front_bottom_pack, (nxt * nzt) * 2, MPI_BYTE, forward, 63, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖in闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != 0){
		MPI_Isend(block_info->halo_Txx_left_bottom_pack, (nyt * nzt) * 2, MPI_BYTE, left, 64, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_ss_left_bottom_pack, (nyt * nzt) * 2, MPI_BYTE, left, 65, MCW, &req_send[(*send_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != Xmax - 1){
		MPI_Isend(block_info->halo_Txy_right_bottom_pack, (nyt * nzt) * 2, MPI_BYTE, right, 66, MCW, &req_send[(*send_count)++]);
		MPI_Isend(block_info->halo_Txz_right_bottom_pack, (nyt * nzt) * 2, MPI_BYTE, right, 67, MCW, &req_send[(*send_count)++]);
	}
	
	
}

void Recv_halo_bottom_T(int myid, int *coord, blockInfo *block_info, MPI_Request *req_send, MPI_Request *req_recv, int *send_count, int *recv_count, int up, int down, int back, int forward, int left, int right, MPI_Comm MCW, int nxt, int nyt, int nzt){

	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;
	//闁兼眹鍎扮粭澶愬及閻ョ偓�???€涙ɑ浠樺☉鎾愁儔濞间即鎯冮崟顔剧?缂佸?�???
	if(coord[2] != 0){
		MPI_Irecv(block_info->halo_Txz_up_recv, (nxt * nyt) * 2, MPI_BYTE, down, 19, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Tyz_up_recv, (nxt * nyt) * 2, MPI_BYTE, down, 20, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榶閺夌偟顥愮粩鐔虹磽濮掝槒n闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != 0){
		MPI_Irecv(block_info->halo_Tyz_front_bottom_recv, (nxt * nzt) * 2, MPI_BYTE, back, 62, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Txy_front_bottom_recv, (nxt * nzt) * 2, MPI_BYTE, back, 63, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ヨ姤�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[1] != Ymax - 1){
		MPI_Irecv(block_info->halo_Tyy_back_bottom_recv, (nxt * nzt) * 2, MPI_BYTE, forward, 60, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_ss_back_bottom_recv, (nxt * nzt) * 2, MPI_BYTE, forward, 61, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖in闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != 0){
		MPI_Irecv(block_info->halo_Txy_right_bottom_recv, (nyt * nzt) * 2, MPI_BYTE, left, 66, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_Txz_right_bottom_recv, (nyt * nzt) * 2, MPI_BYTE, left, 67, MCW, &req_recv[(*recv_count)++]);
	}
	//闁兼眹鍎扮粭澶愬及閻ョ粯�???壕�??彾缂傚�?�螖ax闁汇劌�??换妯肩矙閿燂�??
	if(coord[0] != Xmax - 1){
		MPI_Irecv(block_info->halo_Txx_left_bottom_recv, (nyt * nzt) * 2, MPI_BYTE, right, 64, MCW, &req_recv[(*recv_count)++]);
		MPI_Irecv(block_info->halo_ss_left_bottom_recv, (nyt * nzt) * 2, MPI_BYTE, right, 65, MCW, &req_recv[(*recv_count)++]);
	}
}
//-------------------------------------------------------------------------------------------------------------------------------
//wait data send and recive
void Wait_halo_send(MPI_Request *req_send, int send_count, MPI_Request *req_recv, int recv_count){
    MPI_Status status;
    // �ȴ����з��Ͳ������
    for (int i = 0; i < send_count; ++i) {
        MPI_Wait(&req_send[i], &status);
    }
}

void Wait_halo_recv(MPI_Request *req_send, int send_count, MPI_Request *req_recv, int recv_count){
    MPI_Status status;
    // �ȴ����н��ղ������
    for (int i = 0; i < recv_count; ++i) {
        MPI_Wait(&req_recv[i], &status);
    }
}

void Pack_haloVtop(int pid, int halo_part, int *coord, blockInfo *block_info, int nxt, int nyt, int nzt, cudaStream_t stream){
	
	dim3 Block;
	dim3 Grid;
	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;


	//闁瑰灚鎸哥€垫Φy妤犵偞濞�???浼存儍閸曨剚娈堕柟璇″櫙缁辨繈寮甸埀顒併亜閺堢數鐟愰柣銊ュⅰy
	Block.x = 16;
	Block.y = 8;
	Block.z = 1;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = 1;

	//闁兼眹鍎扮粭澶愬及閻ョ穲ax閺夆晜绋撻埢鍏肩▔閺冣偓濡茬珦alo 2  閻犱緤绱曢悾�???
	if(coord[2] != Zmax - 1 && halo_part == 2){
		gather_halo_xy <<< Grid, Block, 0, stream >>>(block_info->d_vuz, block_info->halo_Vuz_up_pack, nzt - 2, halo_part);  //tran
		gather_halo_xy <<< Grid, Block, 0, stream >>>(block_info->d_vwz, block_info->halo_Vwz_up_pack, nzt - 2, halo_part);
	}

	//闁瑰灚鎸哥€垫Φz妤犵偞濞�???浼存儍閸曨剚娈堕柟鐧告嫹
	Block.x = 16;
	Block.y = 1;
	Block.z = 8;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = 1;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	//闁兼眹鍎扮粭澶愬及閻モ偓min閺夆晜绋撻埢�???,闁瑰灚鎸哥€垫﹢寮甸�?顒勫礈�?�ュ?妗ㄩ柣銊ュⅰz妤犵偞濞�???�???
	if(coord[1] != 0)
	{
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vux, block_info->halo_Vux_back_top_pack, tran, halo_part);
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vuz, block_info->halo_Vuz_back_top_pack, tran, halo_part);
	}
	//闁兼眹鍎扮粭澶愬及閻モ偓max閺夆晜绋撻埢濂告晬鐏炴儳鈪甸柛鏍ф噺濞撳爼宕ユ惔銊︽�?�闁汇劌�???妤犵偞濞�???�???
	if(coord[1] != Ymax - 1)
	{
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vuy, block_info->halo_Vuy_front_top_pack, nyt - 2, halo_part);
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vwy, block_info->halo_Vwy_front_top_pack, nyt - 2, halo_part);
	}

	//闁瑰灚鎸哥€垫Χz妤犵偞濞�???浼存儍閸曨剚娈堕柟鐧告嫹
	Block.x = 1;
	Block.y = 16;
	Block.z = 8;
	Grid.x = 1;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榅min闁汇劌�??换妯肩矙鐎ｅ墎绀夐柟�???尭鐎垫﹢寮甸�?顒€顔忛敃浣虹彾闁汇劌澧?妤犵偞濞�???�???
	if(coord[0] != 0){
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vuy, block_info->halo_Vuy_left_top_pack, tran, halo_part);
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vuz, block_info->halo_Vuz_left_top_pack, tran, halo_part);
	}
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榅max闁汇劌�??换妯肩矙鐎ｅ墎绀夐柟�???尭鐎垫﹢寮甸�?顒勫矗鐎圭姷鐝堕柣銊ュⅱz妤犵偞濞�???�???
	if(coord[0] != Xmax - 1){
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vux, block_info->halo_Vux_right_top_pack, nxt - 2, halo_part);
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vwx, block_info->halo_Vwx_right_top_pack, nxt - 2, halo_part);
	}

}

void Pack_haloVmid(int pid, int halo_part, int *coord, blockInfo *block_info, int nxt, int nyt, int nzt, cudaStream_t stream){
	
	dim3 Block;
	dim3 Grid;
	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;

	//闁瑰灚鎸哥€垫Φz妤犵偞濞�???浼存儍閸曨剚娈堕柟鐧告嫹
	Block.x = 16;
	Block.y = 1;
	Block.z = 8;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = 1;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	//闁兼眹鍎扮粭澶愬及閻モ偓min閺夆晜绋撻埢�???,闁瑰灚鎸哥€垫﹢寮甸�?顒勫礈�?�ュ?妗ㄩ柣銊ュⅰz妤犵偞濞�???�???
	if(coord[1] != 0)
	{
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vux, block_info->halo_Vux_back_mid_pack, tran, halo_part);
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vuz, block_info->halo_Vuz_back_mid_pack, tran, halo_part);
	}
	//闁兼眹鍎扮粭澶愬及閻モ偓max閺夆晜绋撻埢濂告晬鐏炴儳鈪甸柛鏍ф噺濞撳爼宕ユ惔銊︽�?�闁汇劌�???妤犵偞濞�???�???
	if(coord[1] != Ymax - 1)
	{
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vuy, block_info->halo_Vuy_front_mid_pack, nyt - 2, halo_part);
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vwy, block_info->halo_Vwy_front_mid_pack, nyt - 2, halo_part);
	}

	//闁瑰灚鎸哥€垫Χz妤犵偞濞�???浼存儍閸曨剚娈堕柟鐧告嫹
	Block.x = 1;
	Block.y = 16;
	Block.z = 8;
	Grid.x = 1;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榅min闁汇劌�??换妯肩矙鐎ｅ墎绀夐柟�???尭鐎垫﹢寮甸�?顒€顔忛敃浣虹彾闁汇劌澧?妤犵偞濞�???�???
	if(coord[0] != 0){
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vuy, block_info->halo_Vuy_left_mid_pack, tran, halo_part);
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vuz, block_info->halo_Vuz_left_mid_pack, tran, halo_part);
	}
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榅max闁汇劌�??换妯肩矙鐎ｅ墎绀夐柟�???尭鐎垫﹢寮甸�?顒勫矗鐎圭姷鐝堕柣銊ュⅱz妤犵偞濞�???�???
	if(coord[0] != Xmax - 1){
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vux, block_info->halo_Vux_right_mid_pack, nxt - 2, halo_part);
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vwx, block_info->halo_Vwx_right_mid_pack, nxt - 2, halo_part);
	}
}

void Pack_haloVbottom(int pid, int halo_part, int *coord, blockInfo *block_info, int nxt, int nyt, int nzt, cudaStream_t stream){
	
	dim3 Block;
	dim3 Grid;
	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;
	

	//闁瑰灚鎸哥€垫Φz妤犵偞濞�???浼存儍閸曨剚娈堕柟鐧告嫹
	Block.x = 16;
	Block.y = 1;
	Block.z = 8;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = 1;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	//闁兼眹鍎扮粭澶愬及閻モ偓min閺夆晜绋撻埢�???,闁瑰灚鎸哥€垫﹢寮甸�?顒勫礈�?�ュ?妗ㄩ柣銊ュⅰz妤犵偞濞�???�???
	if(coord[1] != 0)
	{
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vux, block_info->halo_Vux_back_bottom_pack, tran, halo_part);
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vuz, block_info->halo_Vuz_back_bottom_pack, tran, halo_part);
	}
	//闁兼眹鍎扮粭澶愬及閻モ偓max閺夆晜绋撻埢濂告晬鐏炴儳鈪甸柛鏍ф噺濞撳爼宕ユ惔銊︽�?�闁汇劌�???妤犵偞濞�???�???
	if(coord[1] != Ymax - 1)
	{
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vuy, block_info->halo_Vuy_front_bottom_pack, nyt - 2, halo_part);
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vwy, block_info->halo_Vwy_front_bottom_pack, nyt - 2, halo_part);
	}

	//闁瑰灚鎸哥€垫Χz妤犵偞濞�???浼存儍閸曨剚娈堕柟鐧告嫹
	Block.x = 1;
	Block.y = 16;
	Block.z = 8;
	Grid.x = 1;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榅min闁汇劌�??换妯肩矙鐎ｅ墎绀夐柟�???尭鐎垫﹢寮甸�?顒€顔忛敃浣虹彾闁汇劌澧?妤犵偞濞�???�???
	if(coord[0] != 0){
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vuy, block_info->halo_Vuy_left_bottom_pack, tran, halo_part);
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vuz, block_info->halo_Vuz_left_bottom_pack, tran, halo_part);
	}
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榅max闁汇劌�??换妯肩矙鐎ｅ墎绀夐柟�???尭鐎垫﹢寮甸�?顒勫矗鐎圭姷鐝堕柣銊ュⅱz妤犵偞濞�???�???
	if(coord[0] != Xmax - 1){
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vux, block_info->halo_Vux_right_bottom_pack, nxt - 2, halo_part);
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vwx, block_info->halo_Vwx_right_bottom_pack, nxt - 2, halo_part);
	}
	
	//闁瑰灚鎸哥€垫Φy妤犵偞濞�???浼存儍閸曨剚娈堕柟璇″櫙缁辨繈寮甸埀顒佺▔鐎ｎ喗�???
	Block.x = 16;
	Block.y = 8;
	Block.z = 1;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = 1;
	if(coord[2] != 0 && halo_part == 0){	
		gather_halo_xy <<< Grid, Block, 0, stream >>>(block_info->d_vux, block_info->halo_Vux_down_pack, tran, halo_part);
		gather_halo_xy <<< Grid, Block, 0, stream >>>(block_info->d_vuy, block_info->halo_Vuy_down_pack, tran, halo_part);
	}

}

void Pack_haloTtop(int pid, int halo_part, int *coord, blockInfo *block_info, int nxt, int nyt, int nzt, cudaStream_t stream){
	
	dim3 Block;
	dim3 Grid;
	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;

	//闁瑰灚鎸哥€垫Φy妤犵偞濞�???浼存儍閸曨剚娈堕柟璇″櫙缁辨繈寮甸埀顒併亜閺堢數鐟愰柣銊ュⅰy
	Block.x = 16;
	Block.y = 8;
	Block.z = 1;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = 1;


	//闁兼眹鍎扮粭澶愬及閻ョ穲ax閺夆晜绋撻埢鍏肩▔閺冣偓濡茬珦alo 0  閻犱緤绱曢悾�???
	if(coord[2] != Zmax - 1 && halo_part == 2){
		gather_halo_xy <<< Grid, Block, 0, stream >>>(block_info->d_txz, block_info->halo_Txz_up_pack, nzt - 2, halo_part);  //tran
		gather_halo_xy <<< Grid, Block, 0, stream >>>(block_info->d_tyz, block_info->halo_Tyz_up_pack, nzt - 2, halo_part);
	}

	//闁瑰灚鎸哥€垫Φz妤犵偞濞�???浼存儍閸曨剚娈堕柟鐧告嫹
	Block.x = 16;
	Block.y = 1;
	Block.z = 8;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = 1;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	//闁兼眹鍎扮粭澶愬及閻モ偓min閺夆晜绋撻埢�???,闁瑰灚鎸哥€垫﹢寮甸�?顒勫礈�?�ュ?妗ㄩ柣銊ュⅰz妤犵偞濞�???�???
	if(coord[1] != 0)
	{
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_tyy, block_info->halo_Tyy_back_top_pack, tran, halo_part);
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_ss, block_info->halo_ss_back_top_pack, tran, halo_part);
	}
	//闁兼眹鍎扮粭澶愬及閻モ偓max閺夆晜绋撻埢濂告晬鐏炴儳鈪甸柛鏍ф噺濞撳爼宕ユ惔銊︽�?�闁汇劌�???妤犵偞濞�???�???
	if(coord[1] != Ymax - 1)
	{
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_tyz, block_info->halo_Tyz_front_top_pack, nyt - 2, halo_part);
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_txy, block_info->halo_Txy_front_top_pack, nyt - 2, halo_part);
	}

	//闁瑰灚鎸哥€垫Χz妤犵偞濞�???浼存儍閸曨剚娈堕柟鐧告嫹
	Block.x = 1;
	Block.y = 16;
	Block.z = 8;
	Grid.x = 1;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榅min闁汇劌�??换妯肩矙鐎ｅ墎绀夐柟�???尭鐎垫﹢寮甸�?顒€顔忛敃浣虹彾闁汇劌澧?妤犵偞濞�???�???
	if(coord[0] != 0){
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_txx, block_info->halo_Txx_left_top_pack, tran, halo_part);
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_ss, block_info->halo_ss_left_top_pack, tran, halo_part);
	}
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榅max闁汇劌�??换妯肩矙鐎ｅ墎绀夐柟�???尭鐎垫﹢寮甸�?顒勫矗鐎圭姷鐝堕柣銊ュⅱz妤犵偞濞�???�???
	if(coord[0] != Xmax - 1){
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_txy, block_info->halo_Txy_right_top_pack, nxt - 2, halo_part);
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_txz, block_info->halo_Txz_right_top_pack, nxt - 2, halo_part);
	}

}

void Pack_haloTmid(int pid, int halo_part, int *coord, blockInfo *block_info, int nxt, int nyt, int nzt, cudaStream_t stream){
	
	dim3 Block;
	dim3 Grid;
	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;

	//闁瑰灚鎸哥€垫Φz妤犵偞濞�???浼存儍閸曨剚娈堕柟鐧告嫹
	Block.x = 16;
	Block.y = 1;
	Block.z = 8;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = 1;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	//闁兼眹鍎扮粭澶愬及閻モ偓min閺夆晜绋撻埢�???,闁瑰灚鎸哥€垫﹢寮甸�?顒勫礈�?�ュ?妗ㄩ柣銊ュⅰz妤犵偞濞�???�???
	if(coord[1] != 0)
	{
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_tyy, block_info->halo_Tyy_back_mid_pack, tran, halo_part);
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_ss, block_info->halo_ss_back_mid_pack, tran, halo_part);
	}
	//闁兼眹鍎扮粭澶愬及閻モ偓max閺夆晜绋撻埢濂告晬鐏炴儳鈪甸柛鏍ф噺濞撳爼宕ユ惔銊︽�?�闁汇劌�???妤犵偞濞�???�???
	if(coord[1] != Ymax - 1)
	{
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_tyz, block_info->halo_Tyz_front_mid_pack, nyt - 2, halo_part);
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_txy, block_info->halo_Txy_front_mid_pack, nyt - 2, halo_part);
	}

	//闁瑰灚鎸哥€垫Χz妤犵偞濞�???浼存儍閸曨剚娈堕柟鐧告嫹
	Block.x = 1;
	Block.y = 16;
	Block.z = 8;
	Grid.x = 1;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榅min闁汇劌�??换妯肩矙鐎ｅ墎绀夐柟�???尭鐎垫﹢寮甸�?顒€顔忛敃浣虹彾闁汇劌澧?妤犵偞濞�???�???
	if(coord[0] != 0){
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_txx, block_info->halo_Txx_left_mid_pack, tran, halo_part);
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_ss, block_info->halo_ss_left_mid_pack, tran, halo_part);
	}
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榅max闁汇劌�??换妯肩矙鐎ｅ墎绀夐柟�???尭鐎垫﹢寮甸�?顒勫矗鐎圭姷鐝堕柣銊ュⅱz妤犵偞濞�???�???
	if(coord[0] != Xmax - 1){
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_txy, block_info->halo_Txy_right_mid_pack, nxt - 2, halo_part);
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_txz, block_info->halo_Txz_right_mid_pack, nxt - 2, halo_part);
	}
	
}

void Pack_haloTbottom(int pid, int halo_part, int *coord, blockInfo *block_info, int nxt, int nyt, int nzt, cudaStream_t stream){
	
	dim3 Block;
	dim3 Grid;
	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;

	//闁瑰灚鎸哥€垫Φz妤犵偞濞�???浼存儍閸曨剚娈堕柟鐧告嫹
	Block.x = 16;
	Block.y = 1;
	Block.z = 8;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = 1;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	//闁兼眹鍎扮粭澶愬及閻モ偓min閺夆晜绋撻埢�???,闁瑰灚鎸哥€垫﹢寮甸�?顒勫礈�?�ュ?妗ㄩ柣銊ュⅰz妤犵偞濞�???�???
	if(coord[1] != 0)
	{
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_tyy, block_info->halo_Tyy_back_bottom_pack, tran, halo_part);
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_ss, block_info->halo_ss_back_bottom_pack, tran, halo_part);
	}
	//闁兼眹鍎扮粭澶愬及閻モ偓max閺夆晜绋撻埢濂告晬鐏炴儳鈪甸柛鏍ф噺濞撳爼宕ユ惔銊︽�?�闁汇劌�???妤犵偞濞�???�???
	if(coord[1] != Ymax - 1)
	{
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_tyz, block_info->halo_Tyz_front_bottom_pack, nyt - 2, halo_part);
		gather_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_txy, block_info->halo_Txy_front_bottom_pack, nyt - 2, halo_part);
	}

	//闁瑰灚鎸哥€垫Χz妤犵偞濞�???浼存儍閸曨剚娈堕柟鐧告嫹
	Block.x = 1;
	Block.y = 16;
	Block.z = 8;
	Grid.x = 1;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榅min闁汇劌�??换妯肩矙鐎ｅ墎绀夐柟�???尭鐎垫﹢寮甸�?顒€顔忛敃浣虹彾闁汇劌澧?妤犵偞濞�???�???
	if(coord[0] != 0){
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_txx, block_info->halo_Txx_left_bottom_pack, tran, halo_part);
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_ss, block_info->halo_ss_left_bottom_pack, tran, halo_part);
	}
	//闁兼眹鍎扮粭澶愬及椤栨稒浠榅max闁汇劌�??换妯肩矙鐎ｅ墎绀夐柟�???尭鐎垫﹢寮甸�?顒勫矗鐎圭姷鐝堕柣銊ュⅱz妤犵偞濞�???�???
	if(coord[0] != Xmax - 1){
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_txy, block_info->halo_Txy_right_bottom_pack, nxt - 2, halo_part);
		gather_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_txz, block_info->halo_Txz_right_bottom_pack, nxt - 2, halo_part);
	}
	
	//闁瑰灚鎸哥€垫Φy妤犵偞濞�???浼存儍閸曨剚娈堕柟璇″櫙缁辨繈寮甸埀顒佺▔鐎ｎ喗�???
	Block.x = 16;
	Block.y = 8;
	Block.z = 1;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = 1;
	if(coord[2] != 0 && halo_part == 0){	
		gather_halo_xy <<< Grid, Block, 0, stream >>>(block_info->d_tzz, block_info->halo_Tzz_down_pack, tran, halo_part);
		gather_halo_xy <<< Grid, Block, 0, stream >>>(block_info->d_ss, block_info->halo_ss_down_pack, tran, halo_part);
	}

}

void Unpack_haloVtop(int halo_part, int *coord, blockInfo *block_info, int nxt, int nyt, int nzt, cudaStream_t stream){
	
	dim3 Block;
	dim3 Grid;
	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;


	Block.x = 16;
	Block.y = 8;
	Block.z = 1;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = 1;

	//闁兼眹鍎扮粭澶愬及閻ョ穲ax閺夆晜绋撻埢鍏肩▔閺冣偓濡茬珦alo 0  閻犱緤绱曢悾�???
	if(coord[2] != Zmax - 1 && halo_part == 2){	
		scatter_halo_xy<<< Grid, Block, 0, stream >>>(block_info->d_vux, block_info->halo_Vux_down_recv, nzt - 1, halo_part);
		scatter_halo_xy<<< Grid, Block, 0, stream >>>(block_info->d_vuy, block_info->halo_Vuy_down_recv, nzt - 1, halo_part);
	}

	Block.x = 16;
	Block.y = 1;
	Block.z = 8;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = 1;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	//闁兼眹鍎扮粭澶愬及閻モ偓min閺夆晜绋撻埢�???
	if(coord[1] != 0)
	{
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vuy, block_info->halo_Vuy_front_top_recv, tran - 1, halo_part);
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vwy, block_info->halo_Vwy_front_top_recv, tran - 1, halo_part);
	}
	//闁兼眹鍎扮粭澶愬及閻モ偓max閺夆晜绋撻埢�???
	if(coord[1] != Ymax - 1)
	{
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vux, block_info->halo_Vux_back_top_recv, nyt - 1, halo_part);
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vuz, block_info->halo_Vuz_back_top_recv, nyt - 1, halo_part);
	}

	Block.x = 1;
	Block.y = 16;
	Block.z = 8;
	Grid.x = 1;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	if(coord[0] != 0){
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vux, block_info->halo_Vux_right_top_recv, tran - 1, halo_part);
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vwx, block_info->halo_Vwx_right_top_recv, tran - 1, halo_part);
	}
	if(coord[0] != Xmax - 1){
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vuy, block_info->halo_Vuy_left_top_recv, nxt - 1, halo_part);
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vuz, block_info->halo_Vuz_left_top_recv, nxt - 1, halo_part);
	}
	


}

void Unpack_haloVmid(int halo_part, int *coord, blockInfo *block_info, int nxt, int nyt, int nzt, cudaStream_t stream){
	
	dim3 Block;
	dim3 Grid;
	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;

	Block.x = 16;
	Block.y = 1;
	Block.z = 8;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = 1;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	//闁兼眹鍎扮粭澶愬及閻モ偓min閺夆晜绋撻埢�???
	if(coord[1] != 0)
	{
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vuy, block_info->halo_Vuy_front_mid_recv, tran - 1, halo_part);
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vwy, block_info->halo_Vwy_front_mid_recv, tran - 1, halo_part);
	}
	//闁兼眹鍎扮粭澶愬及閻モ偓max閺夆晜绋撻埢�???
	if(coord[1] != Ymax - 1)
	{
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vux, block_info->halo_Vux_back_mid_recv, nyt - 1, halo_part);
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vuz, block_info->halo_Vuz_back_mid_recv, nyt - 1, halo_part);
	}

	Block.x = 1;
	Block.y = 16;
	Block.z = 8;
	Grid.x = 1;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	if(coord[0] != 0){
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vux, block_info->halo_Vux_right_mid_recv, tran - 1, halo_part);
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vwx, block_info->halo_Vwx_right_mid_recv, tran - 1, halo_part);
	}
	if(coord[0] != Xmax - 1){
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vuy, block_info->halo_Vuy_left_mid_recv, nxt - 1, halo_part);
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vuz, block_info->halo_Vuz_left_mid_recv, nxt - 1, halo_part);
	}
	
}

void Unpack_haloVbottom(int halo_part, int *coord, blockInfo *block_info, int nxt, int nyt, int nzt, cudaStream_t stream){
	
	dim3 Block;
	dim3 Grid;
	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;	


	Block.x = 16;
	Block.y = 1;
	Block.z = 8;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = 1;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	//闁兼眹鍎扮粭澶愬及閻モ偓min閺夆晜绋撻埢�???
	if(coord[1] != 0)
	{
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vuy, block_info->halo_Vuy_front_bottom_recv, tran - 1, halo_part);
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vwy, block_info->halo_Vwy_front_bottom_recv, tran - 1, halo_part);
	}
	//闁兼眹鍎扮粭澶愬及閻モ偓max閺夆晜绋撻埢�???
	if(coord[1] != Ymax - 1)
	{
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vux, block_info->halo_Vux_back_bottom_recv, nyt - 1, halo_part);
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_vuz, block_info->halo_Vuz_back_bottom_recv, nyt - 1, halo_part);
	}

	Block.x = 1;
	Block.y = 16;
	Block.z = 8;
	Grid.x = 1;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	if(coord[0] != 0){
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vux, block_info->halo_Vux_right_bottom_recv, tran - 1, halo_part);
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vwx, block_info->halo_Vwx_right_bottom_recv, tran - 1, halo_part);
	}
	if(coord[0] != Xmax - 1){
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vuy, block_info->halo_Vuy_left_bottom_recv, nxt - 1, halo_part);
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_vuz, block_info->halo_Vuz_left_bottom_recv, nxt - 1, halo_part);
	}

    Block.x = 16;
	Block.y = 8;
	Block.z = 1;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = 1;
	if(coord[2] != 0 && halo_part == 0){	
		scatter_halo_xy<<< Grid, Block, 0, stream >>>(block_info->d_vuz, block_info->halo_Vuz_up_recv, tran - 1, halo_part);
		scatter_halo_xy<<< Grid, Block, 0, stream >>>(block_info->d_vwz, block_info->halo_Vwz_up_recv, tran - 1, halo_part);
	}

}

void Unpack_haloTtop(int halo_part, int *coord, blockInfo *block_info, int nxt, int nyt, int nzt, cudaStream_t stream){
	
	dim3 Block;
	dim3 Grid;
	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;

    Block.x = 16;
	Block.y = 8;
	Block.z = 1;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = 1;
	//闁兼眹鍎扮粭澶愬及閻ョ穲ax閺夆晜绋撻埢鍏肩▔閺冣偓濡茬珦alo 0  閻犱緤绱曢悾�???
	if(coord[2] != Zmax - 1 && halo_part == 2){	
		scatter_halo_xy<<< Grid, Block, 0, stream >>>(block_info->d_tzz, block_info->halo_Tzz_down_recv, nzt - 1, halo_part);
		scatter_halo_xy<<< Grid, Block, 0, stream >>>(block_info->d_ss, block_info->halo_ss_down_recv,  nzt - 1, halo_part);
	}

	Block.x = 16;
	Block.y = 1;
	Block.z = 8;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = 1;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	//闁兼眹鍎扮粭澶愬及閻モ偓min閺夆晜绋撻埢�???
	if(coord[1] != 0)
	{
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_tyz, block_info->halo_Tyz_front_top_recv, tran - 1, halo_part);
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_txy, block_info->halo_Txy_front_top_recv, tran - 1, halo_part);
	}
	//闁兼眹鍎扮粭澶愬及閻モ偓max閺夆晜绋撻埢�???
	if(coord[1] != Ymax - 1)
	{
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_tyy, block_info->halo_Tyy_back_top_recv, nyt - 1, halo_part);
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_ss, block_info->halo_ss_back_top_recv,  nyt - 1, halo_part);
	}

	Block.x = 1;
	Block.y = 16;
	Block.z = 8;
	Grid.x = 1;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	if(coord[0] != 0){
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_txy, block_info->halo_Txy_right_top_recv, tran - 1, halo_part);
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_txz, block_info->halo_Txz_right_top_recv, tran - 1, halo_part);
	}
	if(coord[0] != Xmax - 1){
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_txx, block_info->halo_Txx_left_top_recv, nxt - 1, halo_part);
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_ss, block_info->halo_ss_left_top_recv,  nxt - 1, halo_part);
	}
	

}

void Unpack_haloTmid(int halo_part, int *coord, blockInfo *block_info, int nxt, int nyt, int nzt, cudaStream_t stream){
	
	dim3 Block;
	dim3 Grid;
	int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;

	Block.x = 16;
	Block.y = 1;
	Block.z = 8;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = 1;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	//闁兼眹鍎扮粭澶愬及閻モ偓min閺夆晜绋撻埢�???
	if(coord[1] != 0)
	{
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_tyz, block_info->halo_Tyz_front_mid_recv, tran - 1, halo_part);
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_txy, block_info->halo_Txy_front_mid_recv, tran - 1, halo_part);
	}
	//闁兼眹鍎扮粭澶愬及閻モ偓max閺夆晜绋撻埢�???
	if(coord[1] != Ymax - 1)
	{
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_tyy, block_info->halo_Tyy_back_mid_recv, nyt - 1, halo_part);
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_ss, block_info->halo_ss_back_mid_recv,  nyt - 1, halo_part);
	}

	Block.x = 1;
	Block.y = 16;
	Block.z = 8;
	Grid.x = 1;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	if(coord[0] != 0){
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_txy, block_info->halo_Txy_right_mid_recv, tran - 1, halo_part);
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_txz, block_info->halo_Txz_right_mid_recv, tran - 1, halo_part);
	}
	if(coord[0] != Xmax - 1){
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_txx, block_info->halo_Txx_left_mid_recv, nxt - 1, halo_part);
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_ss, block_info->halo_ss_left_mid_recv,  nxt - 1, halo_part);
	}
	


}

void Unpack_haloTbottom(int halo_part, int *coord, blockInfo *block_info, int nxt, int nyt, int nzt, cudaStream_t stream){
	
	dim3 Block;
	dim3 Grid;
    int Xmax = PX;
	int Ymax = PY;
	int Zmax = PZ;

	Block.x = 16;
	Block.y = 1;
	Block.z = 8;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = 1;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	//闁兼眹鍎扮粭澶愬及閻モ偓min閺夆晜绋撻埢�???
	if(coord[1] != 0)
	{
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_tyz, block_info->halo_Tyz_front_bottom_recv, tran - 1, halo_part);
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_txy, block_info->halo_Txy_front_bottom_recv, tran - 1, halo_part);
	}
	//闁兼眹鍎扮粭澶愬及閻モ偓max閺夆晜绋撻埢�???
	if(coord[1] != Ymax - 1)
	{
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_tyy, block_info->halo_Tyy_back_bottom_recv, nyt - 1, halo_part);
		scatter_halo_xz <<< Grid, Block, 0, stream >>>(block_info->d_ss, block_info->halo_ss_back_bottom_recv,  nyt - 1, halo_part);
	}

	Block.x = 1;
	Block.y = 16;
	Block.z = 8;
	Grid.x = 1;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = (local_z + Block.z - 1)/ Block.z;
	if(coord[0] != 0){
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_txy, block_info->halo_Txy_right_bottom_recv, tran - 1, halo_part);
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_txz, block_info->halo_Txz_right_bottom_recv, tran - 1, halo_part);
	}
	if(coord[0] != Xmax - 1){
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_txx, block_info->halo_Txx_left_bottom_recv, nxt - 1, halo_part);
		scatter_halo_yz <<< Grid, Block, 0, stream >>>(block_info->d_ss, block_info->halo_ss_left_bottom_recv,  nxt - 1, halo_part);
	}

    Block.x = 16;
	Block.y = 8;
	Block.z = 1;
	Grid.x = (local_x + Block.x - 1)/ Block.x;
	Grid.y = (local_y + Block.y - 1)/ Block.y;
	Grid.z = 1;
	if(coord[2] != 0 && halo_part == 0){	
		scatter_halo_xy<<< Grid, Block, 0, stream >>>(block_info->d_txz, block_info->halo_Txz_up_recv, tran - 1, halo_part);
		scatter_halo_xy<<< Grid, Block, 0, stream >>>(block_info->d_tyz, block_info->halo_Tyz_up_recv, tran - 1, halo_part);
	}

}
//*************************************************************************************************************************
//*************************************************************************************************************************
void callCUDAKernel(blockInfo *parameter, int nxt_pro, int nyt_pro, int nzt_pro, float DT, float F0, int sn, int *coord, int myid, int up, int down, int back, int forward, int left, int right, MPI_Comm MCW, int sxnew, int synew, int sznew, int NT)
{
	float T0 = 1.2 / F0;
	int ix, iy, iz;
	int top = 2;
	int mid = 1;
	int bottom = 0;
	//every part size
	int nxt = nxt_pro;
	int nyt = nyt_pro;
	int nzt = nzt_pro / 3;
	//V and T kernel function block size and grid size
	dim3 Blocksize = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	dim3 Gridsize = dim3(divUp(nxt, BLOCK_SIZE_X), divUp(nyt, BLOCK_SIZE_Y), divUp(nzt, BLOCK_SIZE_Z));
	//Source kernel function block size and grid size
	dim3 BlocksizeSource = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	dim3 GridsizeSource = dim3(divUp(nxt_pro, BLOCK_SIZE_X), divUp(nyt_pro, BLOCK_SIZE_Y), divUp(nzt_pro, BLOCK_SIZE_Z));
	//cudaMalloc for halo
	size_t mem_sizexoy = nxt * nyt * sizeof(half);
	size_t mem_sizexoz = nxt * nzt * sizeof(half);
	size_t mem_sizeyoz = nyt * nzt * sizeof(half);
	//output parameter
	float **sis_x; 
	float **sis_x2;
	float **sis_x3;
	float ***txx3d;
	float ***StressParameterxz;
	float ***ddzk;
	sis_x = space2d(nzt_pro, NT);
	sis_x2 = space2d(nzt_pro, NT);
	sis_x3 = space2d(nzt_pro, NT);
	txx3d = space3d(nzt_pro, nyt_pro, nxt_pro);
	StressParameterxz = space3d(nzt_pro, nyt_pro, nxt_pro);
	ddzk = space3d(nzt_pro, nyt_pro, nxt_pro);
	half *h_StressParameterxz = (half *)calloc(nzt_pro * nyt_pro * nxt_pro, sizeof(half));
	half *h_txx = (half *)calloc(nzt_pro * nyt_pro * nxt_pro, sizeof(half));
	half *h_dzk = (half *)calloc(nzt_pro * nyt_pro * nxt_pro, sizeof(half));
	size_t mem_sizeHalf = nzt_pro * nyt_pro * nxt_pro * sizeof(half);
	//convey V halo size
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vux_down_pack), mem_sizexoy));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuy_down_pack), mem_sizexoy));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuz_up_pack), mem_sizexoy));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vwz_up_pack), mem_sizexoy));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txz_up_pack), mem_sizexoy));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Tyz_up_pack), mem_sizexoy));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Tzz_down_pack), mem_sizexoy));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_ss_down_pack), mem_sizexoy));
	//top
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vux_right_top_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vux_back_top_pack), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuy_left_top_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuy_front_top_pack), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuz_left_top_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuz_back_top_pack), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vwx_right_top_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vwy_front_top_pack), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txx_left_top_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txy_right_top_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txy_front_top_pack), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txz_right_top_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Tyy_back_top_pack), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Tyz_front_top_pack), mem_sizexoz));
  	CUDA_CHECK(cudaMalloc(&(parameter->halo_ss_left_top_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_ss_back_top_pack), mem_sizexoz));
	//mid
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vux_right_mid_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vux_back_mid_pack), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuy_left_mid_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuy_front_mid_pack), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuz_left_mid_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuz_back_mid_pack), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vwx_right_mid_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vwy_front_mid_pack), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txx_left_mid_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txy_right_mid_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txy_front_mid_pack), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txz_right_mid_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Tyy_back_mid_pack), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Tyz_front_mid_pack), mem_sizexoz));
  	CUDA_CHECK(cudaMalloc(&(parameter->halo_ss_left_mid_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_ss_back_mid_pack), mem_sizexoz));
	//bottom
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vux_right_bottom_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vux_back_bottom_pack), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuy_left_bottom_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuy_front_bottom_pack), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuz_left_bottom_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuz_back_bottom_pack), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vwx_right_bottom_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vwy_front_bottom_pack), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txx_left_bottom_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txy_right_bottom_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txy_front_bottom_pack), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txz_right_bottom_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Tyy_back_bottom_pack), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Tyz_front_bottom_pack), mem_sizexoz));
  	CUDA_CHECK(cudaMalloc(&(parameter->halo_ss_left_bottom_pack), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_ss_back_bottom_pack), mem_sizexoz));
	//convey T halo size
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuz_up_recv), mem_sizexoy));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vwz_up_recv), mem_sizexoy));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txz_up_recv), mem_sizexoy));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Tyz_up_recv), mem_sizexoy));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vux_down_recv), mem_sizexoy));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuy_down_recv), mem_sizexoy));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Tzz_down_recv), mem_sizexoy));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_ss_down_recv), mem_sizexoy));
	//top
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vux_right_top_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vux_back_top_recv), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuy_left_top_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuy_front_top_recv), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuz_left_top_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuz_back_top_recv), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vwx_right_top_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vwy_front_top_recv), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txx_left_top_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txy_right_top_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txy_front_top_recv), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txz_right_top_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Tyy_back_top_recv), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Tyz_front_top_recv), mem_sizexoz));
  	CUDA_CHECK(cudaMalloc(&(parameter->halo_ss_left_top_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_ss_back_top_recv), mem_sizexoz));
	//mid
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vux_right_mid_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vux_back_mid_recv), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuy_left_mid_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuy_front_mid_recv), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuz_left_mid_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuz_back_mid_recv), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vwx_right_mid_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vwy_front_mid_recv), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txx_left_mid_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txy_right_mid_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txy_front_mid_recv), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txz_right_mid_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Tyy_back_mid_recv), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Tyz_front_mid_recv), mem_sizexoz));
  	CUDA_CHECK(cudaMalloc(&(parameter->halo_ss_left_mid_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_ss_back_mid_recv), mem_sizexoz));
	//bottom
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vux_right_bottom_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vux_back_bottom_recv), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuy_left_bottom_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuy_front_bottom_recv), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuz_left_bottom_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vuz_back_bottom_recv), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vwx_right_bottom_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Vwy_front_bottom_recv), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txx_left_bottom_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txy_right_bottom_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txy_front_bottom_recv), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Txz_right_bottom_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Tyy_back_bottom_recv), mem_sizexoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_Tyz_front_bottom_recv), mem_sizexoz));
  	CUDA_CHECK(cudaMalloc(&(parameter->halo_ss_left_bottom_recv), mem_sizeyoz));
	CUDA_CHECK(cudaMalloc(&(parameter->halo_ss_back_bottom_recv), mem_sizexoz));
	cudaDeviceSynchronize();
	// cudaStreamCreate compute;
	cudaStream_t compute_stream;
	cudaStreamCreate(&compute_stream);
	//------------------------------------------------------------------------------------------------------------------
	//NT=0 time 
	int it = 0;
	float tt = it * DT;
	float I_sou;
	//闁告梻濮撮崣鍡�5�7珶閻�?牏鐖?
	if(coord[0] == 1 && coord[1] == 1 && coord[2] == 0){
		printf("it=%d\n", it);
		I_sou = -(1 - 2 * PI * PI * F0 * F0 * (tt - T0) * (tt - T0)) * exp(-PI * PI * F0 * F0 * (tt - T0) * (tt - T0));
		Source<<< GridsizeSource, BlocksizeSource, 0, compute_stream >>>(parameter->d_txx, parameter->d_tyy, parameter->d_tzz, I_sou, sn, nxt_pro, nyt_pro, nzt_pro);
	}
	//tag initial
	int send_count_mid = 0;
	int recv_count_mid = 0;

	int send_countT_mid = 0;
	int recv_countT_mid = 0;

    int send_count_up = 0;
	int recv_count_up = 0;

	int send_countT_up = 0;
	int recv_countT_up = 0;

	int send_count_down = 0;
	int recv_count_down = 0;

	int send_countT_down = 0;
	int recv_countT_down = 0;
	MPI_Request req_halo_mid[MAX_REQUESTS], req_haloT_mid[MAX_REQUESTS], req_halo_up[MAX_REQUESTS], req_haloT_up[MAX_REQUESTS], req_halo_down[MAX_REQUESTS], req_haloT_down[MAX_REQUESTS];
	MPI_Request req_halore_mid[MAX_REQUESTS], req_haloTre_mid[MAX_REQUESTS],req_halore_up[MAX_REQUESTS], req_haloTre_up[MAX_REQUESTS],req_halore_down[MAX_REQUESTS], req_haloTre_down[MAX_REQUESTS];
	//------------------------------------------------------------------------------------------------------------------
	//------------------------------------------------------------------------------------------------------------------
	//computer Vmid
	FD_Vmid<<<Gridsize, Blocksize, 0, compute_stream>>>(parameter->d_vux, parameter->d_vuy, parameter->d_vuz, parameter->d_txx, parameter->d_tyy, parameter->d_tzz, parameter->d_txz, parameter->d_txy, parameter->d_tyz, parameter->d_pmlxSxx, parameter->d_pmlySxy, parameter->d_pmlzSxz, parameter->d_pmlxSxy, parameter->d_pmlySyy, parameter->d_pmlzSyz, parameter->d_pmlxSxz, parameter->d_pmlySyz, parameter->d_pmlzSzz, parameter->d_SXxx, parameter->d_SXxy, parameter->d_SXxz, parameter->d_SYxy, parameter->d_SYyy, parameter->d_SYyz, parameter->d_SZxz, parameter->d_SZyz, parameter->d_SZzz, parameter->d_e_dxi, parameter->d_dxi, parameter->d_e_dxi2, parameter->d_dxi2, parameter->d_e_dyj, parameter->d_dyj, parameter->d_e_dyj2, parameter->d_dyj2, parameter->d_e_dzk, parameter->d_dzk, parameter->d_dzk2, parameter->d_e_dzk2, parameter->d_ss, parameter->d_vwx, parameter->d_vwy, parameter->d_vwz, parameter->d_vwx2, parameter->d_vwy2, parameter->d_vwz2, parameter->d_SXss, parameter->d_SYss, parameter->d_SZss, parameter->d_pmlxss, parameter->d_pmlyss, parameter->d_pmlzss, parameter->d_VelocityWParameter1x, parameter->d_VelocityWParameter1y, parameter->d_VelocityWParameter1z, parameter->d_VelocityWParameter2x, parameter->d_VelocityWParameter2y, parameter->d_VelocityWParameter2z, parameter->d_VelocityWParameter3x, parameter->d_VelocityWParameter3y, parameter->d_VelocityWParameter3z, parameter->d_VelocityUParameter1x, parameter->d_VelocityUParameter1y, parameter->d_VelocityUParameter1z, parameter->d_VelocityUParameter2x, parameter->d_VelocityUParameter2y, parameter->d_VelocityUParameter2z, DT, nxt, nyt, nzt, mid);
	// test kernel function
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
	//pack Vmid
	Pack_haloVmid(myid, mid, coord, parameter, nxt, nyt, nzt, compute_stream);
	//send Vmid
	Send_halo_mid_V(myid, coord, parameter, req_halo_mid, req_halore_mid, &send_count_mid, &recv_count_mid, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
	//computer Vtop
	FD_Vtop<<<Gridsize, Blocksize, 0, compute_stream>>>(parameter->d_vux, parameter->d_vuy, parameter->d_vuz, parameter->d_txx, parameter->d_tyy, parameter->d_tzz, parameter->d_txz, parameter->d_txy, parameter->d_tyz, parameter->d_pmlxSxx, parameter->d_pmlySxy, parameter->d_pmlzSxz, parameter->d_pmlxSxy, parameter->d_pmlySyy, parameter->d_pmlzSyz, parameter->d_pmlxSxz, parameter->d_pmlySyz, parameter->d_pmlzSzz, parameter->d_SXxx, parameter->d_SXxy, parameter->d_SXxz, parameter->d_SYxy, parameter->d_SYyy, parameter->d_SYyz, parameter->d_SZxz, parameter->d_SZyz, parameter->d_SZzz, parameter->d_e_dxi, parameter->d_dxi, parameter->d_e_dxi2, parameter->d_dxi2, parameter->d_e_dyj, parameter->d_dyj, parameter->d_e_dyj2, parameter->d_dyj2, parameter->d_e_dzk, parameter->d_dzk, parameter->d_dzk2, parameter->d_e_dzk2, parameter->d_ss, parameter->d_vwx, parameter->d_vwy, parameter->d_vwz, parameter->d_vwx2, parameter->d_vwy2, parameter->d_vwz2, parameter->d_SXss, parameter->d_SYss, parameter->d_SZss, parameter->d_pmlxss, parameter->d_pmlyss, parameter->d_pmlzss, parameter->d_VelocityWParameter1x, parameter->d_VelocityWParameter1y, parameter->d_VelocityWParameter1z, parameter->d_VelocityWParameter2x, parameter->d_VelocityWParameter2y, parameter->d_VelocityWParameter2z, parameter->d_VelocityWParameter3x, parameter->d_VelocityWParameter3y, parameter->d_VelocityWParameter3z, parameter->d_VelocityUParameter1x, parameter->d_VelocityUParameter1y, parameter->d_VelocityUParameter1z, parameter->d_VelocityUParameter2x, parameter->d_VelocityUParameter2y, parameter->d_VelocityUParameter2z, DT, nxt, nyt, nzt, top); 
	//pack Vtop
	Pack_haloVtop(myid, top, coord, parameter, nxt, nyt, nzt, compute_stream);
	//send Vtop
	Send_halo_top_V(myid, coord, parameter, req_halo_up, req_halore_up, &send_count_up, &recv_count_up, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);


	cudaDeviceSynchronize();
	//computer Vbottom
	FD_Vbottom<<<Gridsize, Blocksize, 0, compute_stream>>>(parameter->d_vux, parameter->d_vuy, parameter->d_vuz, parameter->d_txx, parameter->d_tyy, parameter->d_tzz, parameter->d_txz, parameter->d_txy, parameter->d_tyz, parameter->d_pmlxSxx, parameter->d_pmlySxy, parameter->d_pmlzSxz, parameter->d_pmlxSxy, parameter->d_pmlySyy, parameter->d_pmlzSyz, parameter->d_pmlxSxz, parameter->d_pmlySyz, parameter->d_pmlzSzz, parameter->d_SXxx, parameter->d_SXxy, parameter->d_SXxz, parameter->d_SYxy, parameter->d_SYyy, parameter->d_SYyz, parameter->d_SZxz, parameter->d_SZyz, parameter->d_SZzz, parameter->d_e_dxi, parameter->d_dxi, parameter->d_e_dxi2, parameter->d_dxi2, parameter->d_e_dyj, parameter->d_dyj, parameter->d_e_dyj2, parameter->d_dyj2, parameter->d_e_dzk, parameter->d_dzk, parameter->d_dzk2, parameter->d_e_dzk2, parameter->d_ss, parameter->d_vwx, parameter->d_vwy, parameter->d_vwz, parameter->d_vwx2, parameter->d_vwy2, parameter->d_vwz2, parameter->d_SXss, parameter->d_SYss, parameter->d_SZss, parameter->d_pmlxss, parameter->d_pmlyss, parameter->d_pmlzss, parameter->d_VelocityWParameter1x, parameter->d_VelocityWParameter1y, parameter->d_VelocityWParameter1z, parameter->d_VelocityWParameter2x, parameter->d_VelocityWParameter2y, parameter->d_VelocityWParameter2z, parameter->d_VelocityWParameter3x, parameter->d_VelocityWParameter3y, parameter->d_VelocityWParameter3z, parameter->d_VelocityUParameter1x, parameter->d_VelocityUParameter1y, parameter->d_VelocityUParameter1z, parameter->d_VelocityUParameter2x, parameter->d_VelocityUParameter2y, parameter->d_VelocityUParameter2z, DT, nxt, nyt, nzt, bottom);
	//recv Vmid
	Recv_halo_mid_V(myid, coord, parameter, req_halo_mid, req_halore_mid, &send_count_mid, &recv_count_mid, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
	//pack Vbottom
	Pack_haloVbottom(myid, bottom, coord, parameter, nxt, nyt, nzt, compute_stream);
	//send Vbottom
	Send_halo_bottom_V(myid, coord, parameter, req_halo_down, req_halore_down, &send_count_down, &recv_count_down, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
	
	

	//wait Vmid
    Wait_halo_send(req_halo_mid, send_count_mid, req_halore_mid, recv_count_mid);
	Wait_halo_recv(req_halo_mid, send_count_mid, req_halore_mid, recv_count_mid);
	send_count_mid = 0;
	recv_count_mid = 0;
	//unpack Vmid
	Unpack_haloVmid(mid, coord, parameter, nxt, nyt, nzt, compute_stream);

	cudaDeviceSynchronize();
	//computer Tmid
	FD_Tmid<<<Gridsize, Blocksize, 0, compute_stream>>>(parameter->d_vux, parameter->d_vuy, parameter->d_vuz, parameter->d_txx, parameter->d_tzz, parameter->d_tyy, parameter->d_txz, parameter->d_txy, parameter->d_tyz, parameter->d_pmlxVux, parameter->d_pmlyVuy, parameter->d_pmlzVuz, parameter->d_pmlxVuy, parameter->d_pmlyVux, parameter->d_pmlyVuz, parameter->d_pmlzVuy, parameter->d_pmlzVux, parameter->d_pmlxVuz, parameter->d_Vuxx, parameter->d_Vuxy, parameter->d_Vuxz, parameter->d_Vuyx, parameter->d_Vuyy, parameter->d_Vuyz, parameter->d_Vuzx, parameter->d_Vuzy, parameter->d_Vuzz, parameter->d_e_dxi, parameter->d_dxi, parameter->d_e_dxi2, parameter->d_dxi2, parameter->d_e_dyj2, parameter->d_dyj2, parameter->d_e_dyj, parameter->d_dyj, parameter->d_dzk2, parameter->d_e_dzk2, parameter->d_e_dzk, parameter->d_dzk, parameter->d_Vwxx, parameter->d_Vwyy, parameter->d_Vwzz, parameter->d_vwx, parameter->d_vwy, parameter->d_vwz, parameter->d_ss, parameter->d_pmlxVwx, parameter->d_pmlyVwy, parameter->d_pmlzVwz, parameter->d_PressParameter1, parameter->d_PressParameter2, parameter->d_StressParameter1, parameter->d_StressParameter2, parameter->d_StressParameter3, parameter->d_StressParameterxy, parameter->d_StressParameterxz, parameter->d_StressParameteryz, DT, nxt, nyt, nzt, mid);
	//recv Vtop
	Recv_halo_top_V(myid, coord, parameter, req_halo_up, req_halore_up, &send_count_up, &recv_count_up, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
	//pack Tmid
	Pack_haloTmid(myid, mid, coord, parameter, nxt, nyt, nzt, compute_stream);
	//send Tmid
	Send_halo_mid_T(myid, coord, parameter, req_haloT_mid, req_haloTre_mid, &send_countT_mid, &recv_countT_mid, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);


	
	//wait Vtop
    Wait_halo_send(req_halo_up, send_count_up, req_halore_up, recv_count_up);
	Wait_halo_recv(req_halo_up, send_count_up, req_halore_up, recv_count_up);
	send_count_up = 0;
	recv_count_up = 0;
	//unpack Vtop
	Unpack_haloVtop(top, coord, parameter, nxt, nyt, nzt, compute_stream);

	cudaDeviceSynchronize();
    //computer Ttop
	FD_Ttop<<<Gridsize, Blocksize, 0, compute_stream>>>(parameter->d_vux, parameter->d_vuy, parameter->d_vuz, parameter->d_txx, parameter->d_tzz, parameter->d_tyy, parameter->d_txz, parameter->d_txy, parameter->d_tyz, parameter->d_pmlxVux, parameter->d_pmlyVuy, parameter->d_pmlzVuz, parameter->d_pmlxVuy, parameter->d_pmlyVux, parameter->d_pmlyVuz, parameter->d_pmlzVuy, parameter->d_pmlzVux, parameter->d_pmlxVuz, parameter->d_Vuxx, parameter->d_Vuxy, parameter->d_Vuxz, parameter->d_Vuyx, parameter->d_Vuyy, parameter->d_Vuyz, parameter->d_Vuzx, parameter->d_Vuzy, parameter->d_Vuzz, parameter->d_e_dxi, parameter->d_dxi, parameter->d_e_dxi2, parameter->d_dxi2, parameter->d_e_dyj2, parameter->d_dyj2, parameter->d_e_dyj, parameter->d_dyj, parameter->d_dzk2, parameter->d_e_dzk2, parameter->d_e_dzk, parameter->d_dzk, parameter->d_Vwxx, parameter->d_Vwyy, parameter->d_Vwzz, parameter->d_vwx, parameter->d_vwy, parameter->d_vwz, parameter->d_ss, parameter->d_pmlxVwx, parameter->d_pmlyVwy, parameter->d_pmlzVwz, parameter->d_PressParameter1, parameter->d_PressParameter2, parameter->d_StressParameter1, parameter->d_StressParameter2, parameter->d_StressParameter3, parameter->d_StressParameterxy, parameter->d_StressParameterxz, parameter->d_StressParameteryz, DT, nxt, nyt, nzt, top);
	//recv Vbottom
	Recv_halo_bottom_V(myid, coord, parameter, req_halo_down, req_halore_down, &send_count_down, &recv_count_down, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
	//pack Ttop
	Pack_haloTtop(myid, top, coord, parameter, nxt, nyt, nzt, compute_stream);
	//send Ttop
	Send_halo_top_T(myid, coord, parameter, req_haloT_up, req_haloTre_up, &send_countT_up, &recv_countT_up, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
    
	
	
	//wait Vbottom
    Wait_halo_send(req_halo_down, send_count_down, req_halore_down, recv_count_down);
	Wait_halo_recv(req_halo_down, send_count_down, req_halore_down, recv_count_down);
	send_count_down = 0;
	recv_count_down = 0;
	//unpack Vbottom
	Unpack_haloVbottom(bottom, coord, parameter, nxt, nyt, nzt, compute_stream);
	
	cudaDeviceSynchronize();
	//computer Tbottom
	FD_Tbottom<<<Gridsize, Blocksize, 0, compute_stream>>>(parameter->d_vux, parameter->d_vuy, parameter->d_vuz, parameter->d_txx, parameter->d_tzz, parameter->d_tyy, parameter->d_txz, parameter->d_txy, parameter->d_tyz, parameter->d_pmlxVux, parameter->d_pmlyVuy, parameter->d_pmlzVuz, parameter->d_pmlxVuy, parameter->d_pmlyVux, parameter->d_pmlyVuz, parameter->d_pmlzVuy, parameter->d_pmlzVux, parameter->d_pmlxVuz, parameter->d_Vuxx, parameter->d_Vuxy, parameter->d_Vuxz, parameter->d_Vuyx, parameter->d_Vuyy, parameter->d_Vuyz, parameter->d_Vuzx, parameter->d_Vuzy, parameter->d_Vuzz, parameter->d_e_dxi, parameter->d_dxi, parameter->d_e_dxi2, parameter->d_dxi2, parameter->d_e_dyj2, parameter->d_dyj2, parameter->d_e_dyj, parameter->d_dyj, parameter->d_dzk2, parameter->d_e_dzk2, parameter->d_e_dzk, parameter->d_dzk, parameter->d_Vwxx, parameter->d_Vwyy, parameter->d_Vwzz, parameter->d_vwx, parameter->d_vwy, parameter->d_vwz, parameter->d_ss, parameter->d_pmlxVwx, parameter->d_pmlyVwy, parameter->d_pmlzVwz, parameter->d_PressParameter1, parameter->d_PressParameter2, parameter->d_StressParameter1, parameter->d_StressParameter2, parameter->d_StressParameter3, parameter->d_StressParameterxy, parameter->d_StressParameterxz, parameter->d_StressParameteryz, DT, nxt, nyt, nzt, bottom);
	//pack Tbottom
	Pack_haloTbottom(myid, bottom, coord, parameter, nxt, nyt, nzt, compute_stream);



	//output
	CUDA_CHECK(cudaMemcpy(h_txx, parameter->d_txx, mem_sizeHalf, cudaMemcpyDeviceToHost));

	for (iz = 0; iz < nzt_pro; iz++)
	{
		for (iy = 0; iy < nyt_pro; iy++)
		{
			for (ix = 0; ix < nxt_pro; ix++)
			{
				txx3d[iz][iy][ix] = h_txx[iz * nxt_pro * nyt_pro + iy * nxt_pro + ix];
			}
		}
	}
	if (coord[0] == 1 && coord[1] == 1 && coord[2] == 0)
	{
		if (it % 1 == 0)
		{
			for (int sisz = 0; sisz < nzt_pro; sisz++)
			{

				sis_x[sisz][it] = txx3d[sisz][synew][sxnew];
			}
		}
	}
	if (coord[0] == 1 && coord[1] == 1 && coord[2] == 1)
	{
		if (it % 1 == 0)
		{
			for (int sisz = 0; sisz < nzt_pro; sisz++)
			{
				sis_x2[sisz][it] = txx3d[sisz][synew][sxnew];
			}
		}
	}
	if (coord[0] == 1 && coord[1] == 1 && coord[2] == 2)
	{
		if (it % 1 == 0)
		{
			for (int sisz = 0; sisz < nzt_pro; sisz++)
			{
				sis_x3[sisz][it] = txx3d[sisz][synew][sxnew];
			}
		}
	}

	//------------------------------------------------------------------------------------------------------------------
	// 1 ~ NT-1
	for(int it = 1; it < NT - 1; it++)
	{
		if (it % 100 == 0)
		{
			printf("it=%d\n", it);
		}		
		//闁告梻濞€濞撳灝鈹冮敓锟?
		tt = it * DT;
		if(coord[0] == 1 && coord[1] == 1 && coord[2] == 0){
			if (it % MM == 0)
			{
				I_sou = -(1 - 2 * PI * PI * F0 * F0 * (tt - T0) * (tt - T0)) * exp(-PI * PI * F0 * F0 * (tt - T0) * (tt - T0));
				Source << <GridsizeSource, BlocksizeSource>> > (parameter->d_txx, parameter->d_tyy, parameter->d_tzz, I_sou, sn, nxt_pro, nyt_pro, nzt_pro);
				cudaDeviceSynchronize();
			}
		}
		


		//recv Tmid t-1
		Recv_halo_mid_T(myid, coord, parameter, req_haloT_mid, req_haloTre_mid, &send_countT_mid, &recv_countT_mid, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
		//wait Tmid t-1
        Wait_halo_send(req_haloT_mid, send_countT_mid, req_haloTre_mid, recv_countT_mid);
		Wait_halo_recv(req_haloT_mid, send_countT_mid, req_haloTre_mid, recv_countT_mid);
		send_countT_mid = 0;
		recv_countT_mid = 0;
		//send Tbottom t-1
		Send_halo_bottom_T(myid, coord, parameter, req_haloT_down, req_haloTre_down, &send_countT_down, &recv_countT_down, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
		//unpack Tmid t-1
		Unpack_haloTmid(mid, coord, parameter, nxt, nyt, nzt, compute_stream);
		
		cudaDeviceSynchronize();
		//computer Vmid t
		FD_Vmid<<<Gridsize, Blocksize, 0, compute_stream>>>(parameter->d_vux, parameter->d_vuy, parameter->d_vuz, parameter->d_txx, parameter->d_tyy, parameter->d_tzz, parameter->d_txz, parameter->d_txy, parameter->d_tyz, parameter->d_pmlxSxx, parameter->d_pmlySxy, parameter->d_pmlzSxz, parameter->d_pmlxSxy, parameter->d_pmlySyy, parameter->d_pmlzSyz, parameter->d_pmlxSxz, parameter->d_pmlySyz, parameter->d_pmlzSzz, parameter->d_SXxx, parameter->d_SXxy, parameter->d_SXxz, parameter->d_SYxy, parameter->d_SYyy, parameter->d_SYyz, parameter->d_SZxz, parameter->d_SZyz, parameter->d_SZzz, parameter->d_e_dxi, parameter->d_dxi, parameter->d_e_dxi2, parameter->d_dxi2, parameter->d_e_dyj, parameter->d_dyj, parameter->d_e_dyj2, parameter->d_dyj2, parameter->d_e_dzk, parameter->d_dzk, parameter->d_dzk2, parameter->d_e_dzk2, parameter->d_ss, parameter->d_vwx, parameter->d_vwy, parameter->d_vwz, parameter->d_vwx2, parameter->d_vwy2, parameter->d_vwz2, parameter->d_SXss, parameter->d_SYss, parameter->d_SZss, parameter->d_pmlxss, parameter->d_pmlyss, parameter->d_pmlzss, parameter->d_VelocityWParameter1x, parameter->d_VelocityWParameter1y, parameter->d_VelocityWParameter1z, parameter->d_VelocityWParameter2x, parameter->d_VelocityWParameter2y, parameter->d_VelocityWParameter2z, parameter->d_VelocityWParameter3x, parameter->d_VelocityWParameter3y, parameter->d_VelocityWParameter3z, parameter->d_VelocityUParameter1x, parameter->d_VelocityUParameter1y, parameter->d_VelocityUParameter1z, parameter->d_VelocityUParameter2x, parameter->d_VelocityUParameter2y, parameter->d_VelocityUParameter2z, DT, nxt, nyt, nzt, mid);
		//recv Ttop t-1
		Recv_halo_top_T(myid, coord, parameter, req_haloT_up, req_haloTre_up, &send_countT_up, &recv_countT_up, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
		//pack Vmid t
		Pack_haloVmid(myid, mid, coord, parameter, nxt, nyt, nzt, compute_stream);
		//send Vmid t
		Send_halo_mid_V(myid, coord, parameter, req_halo_mid, req_halore_mid, &send_count_mid, &recv_count_mid, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
		
		
		
        //wait Ttop t-1
		Wait_halo_send(req_haloT_up, send_countT_up, req_haloTre_up, recv_countT_up);
		Wait_halo_recv(req_haloT_up, send_countT_up, req_haloTre_up, recv_countT_up);
		send_countT_up = 0;
		recv_countT_up = 0;
		//unpack Ttop t-1
        Unpack_haloTtop(top, coord, parameter, nxt, nyt, nzt, compute_stream);
		
		cudaDeviceSynchronize();
		//computer Vtop t
		FD_Vtop<<<Gridsize, Blocksize, 0, compute_stream>>>(parameter->d_vux, parameter->d_vuy, parameter->d_vuz, parameter->d_txx, parameter->d_tyy, parameter->d_tzz, parameter->d_txz, parameter->d_txy, parameter->d_tyz, parameter->d_pmlxSxx, parameter->d_pmlySxy, parameter->d_pmlzSxz, parameter->d_pmlxSxy, parameter->d_pmlySyy, parameter->d_pmlzSyz, parameter->d_pmlxSxz, parameter->d_pmlySyz, parameter->d_pmlzSzz, parameter->d_SXxx, parameter->d_SXxy, parameter->d_SXxz, parameter->d_SYxy, parameter->d_SYyy, parameter->d_SYyz, parameter->d_SZxz, parameter->d_SZyz, parameter->d_SZzz, parameter->d_e_dxi, parameter->d_dxi, parameter->d_e_dxi2, parameter->d_dxi2, parameter->d_e_dyj, parameter->d_dyj, parameter->d_e_dyj2, parameter->d_dyj2, parameter->d_e_dzk, parameter->d_dzk, parameter->d_dzk2, parameter->d_e_dzk2, parameter->d_ss, parameter->d_vwx, parameter->d_vwy, parameter->d_vwz, parameter->d_vwx2, parameter->d_vwy2, parameter->d_vwz2, parameter->d_SXss, parameter->d_SYss, parameter->d_SZss, parameter->d_pmlxss, parameter->d_pmlyss, parameter->d_pmlzss, parameter->d_VelocityWParameter1x, parameter->d_VelocityWParameter1y, parameter->d_VelocityWParameter1z, parameter->d_VelocityWParameter2x, parameter->d_VelocityWParameter2y, parameter->d_VelocityWParameter2z, parameter->d_VelocityWParameter3x, parameter->d_VelocityWParameter3y, parameter->d_VelocityWParameter3z, parameter->d_VelocityUParameter1x, parameter->d_VelocityUParameter1y, parameter->d_VelocityUParameter1z, parameter->d_VelocityUParameter2x, parameter->d_VelocityUParameter2y, parameter->d_VelocityUParameter2z, DT, nxt, nyt, nzt, top);
		//recv Tbottom t-1
		Recv_halo_bottom_T(myid, coord, parameter, req_haloT_down, req_haloTre_down, &send_countT_down, &recv_countT_down, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
		//pack Vtop t
		Pack_haloVtop(myid, top, coord, parameter, nxt, nyt, nzt, compute_stream);
		//send Vtop t
		Send_halo_top_V(myid, coord, parameter, req_halo_up, req_halore_up, &send_count_up, &recv_count_up, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);


		
        //wait Tbottom t-1
		Wait_halo_send(req_haloT_down, send_countT_down, req_haloTre_down, recv_countT_down);
		Wait_halo_recv(req_haloT_down, send_countT_down, req_haloTre_down, recv_countT_down);
		send_countT_down = 0;
		recv_countT_down = 0;
		//unpack Tbottom t-1
		Unpack_haloTbottom(bottom, coord, parameter, nxt, nyt, nzt, compute_stream);
		
		cudaDeviceSynchronize();
		//computer Vbottom t
		FD_Vbottom<<<Gridsize, Blocksize, 0, compute_stream>>>(parameter->d_vux, parameter->d_vuy, parameter->d_vuz, parameter->d_txx, parameter->d_tyy, parameter->d_tzz, parameter->d_txz, parameter->d_txy, parameter->d_tyz, parameter->d_pmlxSxx, parameter->d_pmlySxy, parameter->d_pmlzSxz, parameter->d_pmlxSxy, parameter->d_pmlySyy, parameter->d_pmlzSyz, parameter->d_pmlxSxz, parameter->d_pmlySyz, parameter->d_pmlzSzz, parameter->d_SXxx, parameter->d_SXxy, parameter->d_SXxz, parameter->d_SYxy, parameter->d_SYyy, parameter->d_SYyz, parameter->d_SZxz, parameter->d_SZyz, parameter->d_SZzz, parameter->d_e_dxi, parameter->d_dxi, parameter->d_e_dxi2, parameter->d_dxi2, parameter->d_e_dyj, parameter->d_dyj, parameter->d_e_dyj2, parameter->d_dyj2, parameter->d_e_dzk, parameter->d_dzk, parameter->d_dzk2, parameter->d_e_dzk2, parameter->d_ss, parameter->d_vwx, parameter->d_vwy, parameter->d_vwz, parameter->d_vwx2, parameter->d_vwy2, parameter->d_vwz2, parameter->d_SXss, parameter->d_SYss, parameter->d_SZss, parameter->d_pmlxss, parameter->d_pmlyss, parameter->d_pmlzss, parameter->d_VelocityWParameter1x, parameter->d_VelocityWParameter1y, parameter->d_VelocityWParameter1z, parameter->d_VelocityWParameter2x, parameter->d_VelocityWParameter2y, parameter->d_VelocityWParameter2z, parameter->d_VelocityWParameter3x, parameter->d_VelocityWParameter3y, parameter->d_VelocityWParameter3z, parameter->d_VelocityUParameter1x, parameter->d_VelocityUParameter1y, parameter->d_VelocityUParameter1z, parameter->d_VelocityUParameter2x, parameter->d_VelocityUParameter2y, parameter->d_VelocityUParameter2z, DT, nxt, nyt, nzt, bottom);
		//recv Vmid t
		Recv_halo_mid_V(myid, coord, parameter, req_halo_mid, req_halore_mid, &send_count_mid, &recv_count_mid, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
		//pack Vbottom t
		Pack_haloVbottom(myid, bottom, coord, parameter, nxt, nyt, nzt, compute_stream);		
		//send Vbottom t
		Send_halo_bottom_V(myid, coord, parameter, req_halo_down, req_halore_down, &send_count_down, &recv_count_down, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);


		
		//wait Vmid t
        Wait_halo_send(req_halo_mid, send_count_mid, req_halore_mid, recv_count_mid);
		Wait_halo_recv(req_halo_mid, send_count_mid, req_halore_mid, recv_count_mid);
		send_count_mid = 0;
		recv_count_mid = 0;
		//unpack Vmid t
		Unpack_haloVmid(mid, coord, parameter, nxt, nyt, nzt, compute_stream);
		
		cudaDeviceSynchronize();
		//computer Tmid t
		FD_Tmid<<<Gridsize, Blocksize, 0, compute_stream>>>(parameter->d_vux, parameter->d_vuy, parameter->d_vuz, parameter->d_txx, parameter->d_tzz, parameter->d_tyy, parameter->d_txz, parameter->d_txy, parameter->d_tyz, parameter->d_pmlxVux, parameter->d_pmlyVuy, parameter->d_pmlzVuz, parameter->d_pmlxVuy, parameter->d_pmlyVux, parameter->d_pmlyVuz, parameter->d_pmlzVuy, parameter->d_pmlzVux, parameter->d_pmlxVuz, parameter->d_Vuxx, parameter->d_Vuxy, parameter->d_Vuxz, parameter->d_Vuyx, parameter->d_Vuyy, parameter->d_Vuyz, parameter->d_Vuzx, parameter->d_Vuzy, parameter->d_Vuzz, parameter->d_e_dxi, parameter->d_dxi, parameter->d_e_dxi2, parameter->d_dxi2, parameter->d_e_dyj2, parameter->d_dyj2, parameter->d_e_dyj, parameter->d_dyj, parameter->d_dzk2, parameter->d_e_dzk2, parameter->d_e_dzk, parameter->d_dzk, parameter->d_Vwxx, parameter->d_Vwyy, parameter->d_Vwzz, parameter->d_vwx, parameter->d_vwy, parameter->d_vwz, parameter->d_ss, parameter->d_pmlxVwx, parameter->d_pmlyVwy, parameter->d_pmlzVwz, parameter->d_PressParameter1, parameter->d_PressParameter2, parameter->d_StressParameter1, parameter->d_StressParameter2, parameter->d_StressParameter3, parameter->d_StressParameterxy, parameter->d_StressParameterxz, parameter->d_StressParameteryz, DT, nxt, nyt, nzt, mid);
		//recv Vtop t
		Recv_halo_top_V(myid, coord, parameter, req_halo_up, req_halore_up, &send_count_up, &recv_count_up, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
		//pack Tmid t
		Pack_haloTmid(myid, mid, coord, parameter, nxt, nyt, nzt, compute_stream);
		//send Tmid t
		Send_halo_mid_T(myid, coord, parameter, req_haloT_mid, req_haloTre_mid, &send_countT_mid, &recv_countT_mid, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);



        // wait Vtop t
        Wait_halo_send(req_halo_up, send_count_up, req_halore_up, recv_count_up);
		Wait_halo_recv(req_halo_up, send_count_up, req_halore_up, recv_count_up);
		send_count_up = 0;
		recv_count_up = 0;
		//unpack Vtop t
        Unpack_haloVtop(top, coord, parameter, nxt, nyt, nzt, compute_stream);
		
		cudaDeviceSynchronize();
		//computer Ttop t
		FD_Ttop<<<Gridsize, Blocksize, 0, compute_stream>>>(parameter->d_vux, parameter->d_vuy, parameter->d_vuz, parameter->d_txx, parameter->d_tzz, parameter->d_tyy, parameter->d_txz, parameter->d_txy, parameter->d_tyz, parameter->d_pmlxVux, parameter->d_pmlyVuy, parameter->d_pmlzVuz, parameter->d_pmlxVuy, parameter->d_pmlyVux, parameter->d_pmlyVuz, parameter->d_pmlzVuy, parameter->d_pmlzVux, parameter->d_pmlxVuz, parameter->d_Vuxx, parameter->d_Vuxy, parameter->d_Vuxz, parameter->d_Vuyx, parameter->d_Vuyy, parameter->d_Vuyz, parameter->d_Vuzx, parameter->d_Vuzy, parameter->d_Vuzz, parameter->d_e_dxi, parameter->d_dxi, parameter->d_e_dxi2, parameter->d_dxi2, parameter->d_e_dyj2, parameter->d_dyj2, parameter->d_e_dyj, parameter->d_dyj, parameter->d_dzk2, parameter->d_e_dzk2, parameter->d_e_dzk, parameter->d_dzk, parameter->d_Vwxx, parameter->d_Vwyy, parameter->d_Vwzz, parameter->d_vwx, parameter->d_vwy, parameter->d_vwz, parameter->d_ss, parameter->d_pmlxVwx, parameter->d_pmlyVwy, parameter->d_pmlzVwz, parameter->d_PressParameter1, parameter->d_PressParameter2, parameter->d_StressParameter1, parameter->d_StressParameter2, parameter->d_StressParameter3, parameter->d_StressParameterxy, parameter->d_StressParameterxz, parameter->d_StressParameteryz, DT, nxt, nyt, nzt, top);
		//recv Vbottom t
		Recv_halo_bottom_V(myid, coord, parameter, req_halo_down, req_halore_down, &send_count_down, &recv_count_down, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
		//pack Ttop t
		Pack_haloTtop(myid, top, coord, parameter, nxt, nyt, nzt, compute_stream);
		//send Ttop t
		Send_halo_top_T(myid, coord, parameter, req_haloT_up, req_haloTre_up, &send_countT_up, &recv_countT_up, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);


		
		//wait Vbottom t
		Wait_halo_send(req_halo_down, send_count_down, req_halore_down, recv_count_down);
		Wait_halo_recv(req_halo_down, send_count_down, req_halore_down, recv_count_down);
		send_count_down = 0;
		recv_count_down = 0;
		//unpack Vbottom t
        Unpack_haloVbottom(bottom, coord, parameter, nxt, nyt, nzt, compute_stream);
		
		cudaDeviceSynchronize();
		//computer Tbottom t
		FD_Tbottom<<<Gridsize, Blocksize, 0, compute_stream>>>(parameter->d_vux, parameter->d_vuy, parameter->d_vuz, parameter->d_txx, parameter->d_tzz, parameter->d_tyy, parameter->d_txz, parameter->d_txy, parameter->d_tyz, parameter->d_pmlxVux, parameter->d_pmlyVuy, parameter->d_pmlzVuz, parameter->d_pmlxVuy, parameter->d_pmlyVux, parameter->d_pmlyVuz, parameter->d_pmlzVuy, parameter->d_pmlzVux, parameter->d_pmlxVuz, parameter->d_Vuxx, parameter->d_Vuxy, parameter->d_Vuxz, parameter->d_Vuyx, parameter->d_Vuyy, parameter->d_Vuyz, parameter->d_Vuzx, parameter->d_Vuzy, parameter->d_Vuzz, parameter->d_e_dxi, parameter->d_dxi, parameter->d_e_dxi2, parameter->d_dxi2, parameter->d_e_dyj2, parameter->d_dyj2, parameter->d_e_dyj, parameter->d_dyj, parameter->d_dzk2, parameter->d_e_dzk2, parameter->d_e_dzk, parameter->d_dzk, parameter->d_Vwxx, parameter->d_Vwyy, parameter->d_Vwzz, parameter->d_vwx, parameter->d_vwy, parameter->d_vwz, parameter->d_ss, parameter->d_pmlxVwx, parameter->d_pmlyVwy, parameter->d_pmlzVwz, parameter->d_PressParameter1, parameter->d_PressParameter2, parameter->d_StressParameter1, parameter->d_StressParameter2, parameter->d_StressParameter3, parameter->d_StressParameterxy, parameter->d_StressParameterxz, parameter->d_StressParameteryz, DT, nxt, nyt, nzt, bottom);
		//pack Tbottom t
		Pack_haloTbottom(myid, bottom, coord, parameter, nxt, nyt, nzt, compute_stream);


			
		//output
		CUDA_CHECK(cudaMemcpy(h_txx, parameter->d_txx, mem_sizeHalf, cudaMemcpyDeviceToHost));
		// CUDA_CHECK(cudaMemcpy(h_StressParameterxz, parameter->d_StressParameterxz, mem_sizeHalf, cudaMemcpyDeviceToHost));
		// CUDA_CHECK(cudaMemcpy(h_dzk, parameter->d_dzk, mem_sizeHalf, cudaMemcpyDeviceToHost));
		for (iz = 0; iz < nzt_pro; iz++)
		{
			for (iy = 0; iy < nyt_pro; iy++)
			{
				for (ix = 0; ix < nxt_pro; ix++)
				{
					txx3d[iz][iy][ix] = h_txx[iz * nxt_pro * nyt_pro + iy * nxt_pro + ix];
					// StressParameterxz[iz][iy][ix] = h_StressParameterxz[iz * nxt_pro * nyt_pro + iy * nxt_pro + ix];
					// ddzk[iz][iy][ix] = h_dzk[iz * nxt_pro * nyt_pro + iy * nxt_pro + ix];
				}
			}
		}

		if (coord[0] == 1 && coord[1] == 1 && coord[2] == 0)
		{
			if (it % 1 == 0)
			{
				for (int sisz = 0; sisz < nzt_pro; sisz++)
				{

					sis_x[sisz][it] = txx3d[sisz][synew][sxnew];
				}
			}
		}
		if (coord[0] == 1 && coord[1] == 1 && coord[2] == 1)
		{
			if (it % 1 == 0)
			{
				for (int sisz = 0; sisz < nzt_pro; sisz++)
				{
					sis_x2[sisz][it] = txx3d[sisz][synew][sxnew];
				}
			}
		}
		if (coord[0] == 1 && coord[1] == 1 && coord[2] == 2)
		{
			if (it % 1 == 0)
			{
				for (int sisz = 0; sisz < nzt_pro; sisz++)
				{
					sis_x3[sisz][it] = txx3d[sisz][synew][sxnew];
				}
			}
		}
		


		//wave snapshort
		// it=1000
		if (myid == 3 && it == 1000)
		{
			char txx3ourceProname[] = "txxpro3 1000.dat";
			wfile3d(txx3ourceProname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 4 && it == 1000)
		{
			char txx4ourceProname[] = "txxpro4 1000.dat";
			wfile3d(txx4ourceProname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 5 && it == 1000)
		{
			char txx5ourceProname[] = "txxpro5 1000.dat";
			wfile3d(txx5ourceProname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 12 && it == 1000)
		{
			char txx12ourceProname[] = "txxpro12 1000.dat";
			wfile3d(txx12ourceProname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 13 && it == 1000)
		{
			char txx13ourceProname[] = "txxpro13 1000.dat";
			wfile3d(txx13ourceProname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 14 && it == 1000)
		{
			char txx14ourceProname[] = "txxpro14 1000.dat";
			wfile3d(txx14ourceProname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 21 && it == 1000)
		{
			char txx21ourceProname[] = "txxpro21 1000.dat";
			wfile3d(txx21ourceProname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 22 && it == 1000)
		{
			char txx22ourceProname[] = "txxpro22 1000.dat";
			wfile3d(txx22ourceProname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 23 && it == 1000)
		{
			char txx23ourceProname[] = "txxpro23 1000.dat";
			wfile3d(txx23ourceProname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		// it=2000
		if (myid == 3 && it == 2000)
		{
			char txx3ource2Proname[] = "txxpro3 2000.dat";
			wfile3d(txx3ource2Proname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 4 && it == 2000)
		{
			char txx4ource2Proname[] = "txxpro4 2000.dat";
			wfile3d(txx4ource2Proname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 5 && it == 2000)
		{
			char txx5ource2Proname[] = "txxpro5 2000.dat";
			wfile3d(txx5ource2Proname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 12 && it == 2000)
		{
			char txx12ource2Proname[] = "txxpro12 2000.dat";
			wfile3d(txx12ource2Proname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 13 && it == 2000)
		{
			char txx13ource2Proname[] = "txxpro13 2000.dat";
			wfile3d(txx13ource2Proname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 14 && it == 2000)
		{
			char txx14ource2Proname[] = "txxpro14 2000.dat";
			wfile3d(txx14ource2Proname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 21 && it == 2000)
		{
			char txx21ource2Proname[] = "txxpro21 2000.dat";
			wfile3d(txx21ource2Proname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 22 && it == 2000)
		{
			char txx22ource2Proname[] = "txxpro22 2000.dat";
			wfile3d(txx22ource2Proname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 23 && it == 2000)
		{
			char txx23ource2Proname[] = "txxpro23 2000.dat";
			wfile3d(txx23ource2Proname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		// it=3000
		if (myid == 3 && it == 3000)
		{
			char txx3ource3Proname[] = "txxpro3 3000.dat";
			wfile3d(txx3ource3Proname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 4 && it == 3000)
		{
			char txx4ource3Proname[] = "txxpro4 3000.dat";
			wfile3d(txx4ource3Proname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 5 && it == 3000)
		{
			char txx5ource3Proname[] = "txxpro5 3000.dat";
			wfile3d(txx5ource3Proname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 12 && it == 3000)
		{
			char txx12ource3Proname[] = "txxpro12 3000.dat";
			wfile3d(txx12ource3Proname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 13 && it == 3000)
		{
			char txx13ource3Proname[] = "txxpro13 3000.dat";
			wfile3d(txx13ource3Proname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 14 && it == 3000)
		{
			char txx14ource3Proname[] = "txxpro14 3000.dat";
			wfile3d(txx14ource3Proname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 21 && it == 3000)
		{
			char txx21ource3Proname[] = "txxpro21 3000.dat";
			wfile3d(txx21ource3Proname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 22 && it == 3000)
		{
			char txx22ource3Proname[] = "txxpro22 3000.dat";
			wfile3d(txx22ource3Proname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}
		if (myid == 23 && it == 3000)
		{
			char txx23ource3Proname[] = "txxpro23 3000.dat";
			wfile3d(txx23ource3Proname, txx3d, nzt_pro, nyt_pro, nxt_pro);
		}

		
	}//end of time loop
	//------------------------------------------------------------------------------------------------------------------
	// NT
	it = NT;
	tt = it * DT;
	if(coord[0] == 1 && coord[1] == 1 && coord[2] == 0){
		printf("it=%d\n", it);
		I_sou = -(1 - 2 * PI * PI * F0 * F0 * (tt - T0) * (tt - T0)) * exp(-PI * PI * F0 * F0 * (tt - T0) * (tt - T0));
		Source<<<GridsizeSource, BlocksizeSource>>>(parameter->d_txx, parameter->d_tyy, parameter->d_tzz, I_sou, sn, nxt_pro, nyt_pro, nzt_pro);
	}



	//recv Tmid NT-1
	Recv_halo_mid_T(myid, coord, parameter, req_haloT_mid, req_haloTre_mid, &send_countT_mid, &recv_countT_mid, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
	//wait Tmid NT-1
	Wait_halo_send(req_haloT_mid, send_countT_mid, req_haloTre_mid, recv_countT_mid);
	Wait_halo_recv(req_haloT_mid, send_countT_mid, req_haloTre_mid, recv_countT_mid);
	send_countT_mid = 0;
	recv_countT_mid = 0;
	//send Tbottom NT-1
	Send_halo_bottom_T(myid, coord, parameter, req_haloT_down, req_haloTre_down, &send_countT_down, &recv_countT_down, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
    //unpack Tmid NT-1
	Unpack_haloTmid(mid, coord, parameter, nxt, nyt, nzt, compute_stream);
	
	cudaDeviceSynchronize();
	//computer Vmid NT
	FD_Vmid<<<Gridsize, Blocksize, 0, compute_stream>>>(parameter->d_vux, parameter->d_vuy, parameter->d_vuz, parameter->d_txx, parameter->d_tyy, parameter->d_tzz, parameter->d_txz, parameter->d_txy, parameter->d_tyz, parameter->d_pmlxSxx, parameter->d_pmlySxy, parameter->d_pmlzSxz, parameter->d_pmlxSxy, parameter->d_pmlySyy, parameter->d_pmlzSyz, parameter->d_pmlxSxz, parameter->d_pmlySyz, parameter->d_pmlzSzz, parameter->d_SXxx, parameter->d_SXxy, parameter->d_SXxz, parameter->d_SYxy, parameter->d_SYyy, parameter->d_SYyz, parameter->d_SZxz, parameter->d_SZyz, parameter->d_SZzz, parameter->d_e_dxi, parameter->d_dxi, parameter->d_e_dxi2, parameter->d_dxi2, parameter->d_e_dyj, parameter->d_dyj, parameter->d_e_dyj2, parameter->d_dyj2, parameter->d_e_dzk, parameter->d_dzk, parameter->d_dzk2, parameter->d_e_dzk2, parameter->d_ss, parameter->d_vwx, parameter->d_vwy, parameter->d_vwz, parameter->d_vwx2, parameter->d_vwy2, parameter->d_vwz2, parameter->d_SXss, parameter->d_SYss, parameter->d_SZss, parameter->d_pmlxss, parameter->d_pmlyss, parameter->d_pmlzss, parameter->d_VelocityWParameter1x, parameter->d_VelocityWParameter1y, parameter->d_VelocityWParameter1z, parameter->d_VelocityWParameter2x, parameter->d_VelocityWParameter2y, parameter->d_VelocityWParameter2z, parameter->d_VelocityWParameter3x, parameter->d_VelocityWParameter3y, parameter->d_VelocityWParameter3z, parameter->d_VelocityUParameter1x, parameter->d_VelocityUParameter1y, parameter->d_VelocityUParameter1z, parameter->d_VelocityUParameter2x, parameter->d_VelocityUParameter2y, parameter->d_VelocityUParameter2z, DT, nxt, nyt, nzt, mid);
	//recv Ttop NT-1
	Recv_halo_top_T(myid, coord, parameter, req_haloT_up, req_haloTre_up, &send_countT_up, &recv_countT_up, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
	//pack Vmid NT
	Pack_haloVmid(myid, mid, coord, parameter, nxt, nyt, nzt, compute_stream);
	//send Vmid NT
	Send_halo_mid_V(myid, coord, parameter, req_halo_mid, req_halore_mid, &send_count_mid, &recv_count_mid, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);


	
	//wait Ttop NT-1
    Wait_halo_send(req_haloT_up, send_countT_up, req_haloTre_up, recv_countT_up);
	Wait_halo_send(req_haloT_up, send_countT_up, req_haloTre_up, recv_countT_up);
	send_countT_up = 0;
	recv_countT_up = 0;
	//unpack Ttop NT-1
    Unpack_haloTtop(top, coord, parameter, nxt, nyt, nzt, compute_stream);
	
	cudaDeviceSynchronize();
	//computer Vtop NT
	FD_Vtop<<<Gridsize, Blocksize, 0, compute_stream>>>(parameter->d_vux, parameter->d_vuy, parameter->d_vuz, parameter->d_txx, parameter->d_tyy, parameter->d_tzz, parameter->d_txz, parameter->d_txy, parameter->d_tyz, parameter->d_pmlxSxx, parameter->d_pmlySxy, parameter->d_pmlzSxz, parameter->d_pmlxSxy, parameter->d_pmlySyy, parameter->d_pmlzSyz, parameter->d_pmlxSxz, parameter->d_pmlySyz, parameter->d_pmlzSzz, parameter->d_SXxx, parameter->d_SXxy, parameter->d_SXxz, parameter->d_SYxy, parameter->d_SYyy, parameter->d_SYyz, parameter->d_SZxz, parameter->d_SZyz, parameter->d_SZzz, parameter->d_e_dxi, parameter->d_dxi, parameter->d_e_dxi2, parameter->d_dxi2, parameter->d_e_dyj, parameter->d_dyj, parameter->d_e_dyj2, parameter->d_dyj2, parameter->d_e_dzk, parameter->d_dzk, parameter->d_dzk2, parameter->d_e_dzk2, parameter->d_ss, parameter->d_vwx, parameter->d_vwy, parameter->d_vwz, parameter->d_vwx2, parameter->d_vwy2, parameter->d_vwz2, parameter->d_SXss, parameter->d_SYss, parameter->d_SZss, parameter->d_pmlxss, parameter->d_pmlyss, parameter->d_pmlzss, parameter->d_VelocityWParameter1x, parameter->d_VelocityWParameter1y, parameter->d_VelocityWParameter1z, parameter->d_VelocityWParameter2x, parameter->d_VelocityWParameter2y, parameter->d_VelocityWParameter2z, parameter->d_VelocityWParameter3x, parameter->d_VelocityWParameter3y, parameter->d_VelocityWParameter3z, parameter->d_VelocityUParameter1x, parameter->d_VelocityUParameter1y, parameter->d_VelocityUParameter1z, parameter->d_VelocityUParameter2x, parameter->d_VelocityUParameter2y, parameter->d_VelocityUParameter2z, DT, nxt, nyt, nzt, top);
	//recv Tbottom NT-1
	Recv_halo_bottom_T(myid, coord, parameter, req_haloT_down, req_haloTre_down, &send_countT_down, &recv_countT_down, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
	//pack Vtop NT
	Pack_haloVtop(myid, top, coord, parameter, nxt, nyt, nzt, compute_stream);
	//send Vtop NT
	Send_halo_top_V(myid, coord, parameter, req_halo_up, req_halore_up, &send_count_up, &recv_count_up, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);



	//wait Tbottom NT-1
	Wait_halo_send(req_haloT_down, send_countT_down, req_haloTre_down, recv_countT_down);
	Wait_halo_send(req_haloT_down, send_countT_down, req_haloTre_down, recv_countT_down);
	send_countT_down = 0;
	recv_countT_down = 0;
	//unpack Tbottom NT-1
    Unpack_haloTbottom(bottom, coord, parameter, nxt, nyt, nzt, compute_stream);
	
	cudaDeviceSynchronize();
	//computer Vbottom NT
	FD_Vbottom<<<Gridsize, Blocksize, 0, compute_stream>>>(parameter->d_vux, parameter->d_vuy, parameter->d_vuz, parameter->d_txx, parameter->d_tyy, parameter->d_tzz, parameter->d_txz, parameter->d_txy, parameter->d_tyz, parameter->d_pmlxSxx, parameter->d_pmlySxy, parameter->d_pmlzSxz, parameter->d_pmlxSxy, parameter->d_pmlySyy, parameter->d_pmlzSyz, parameter->d_pmlxSxz, parameter->d_pmlySyz, parameter->d_pmlzSzz, parameter->d_SXxx, parameter->d_SXxy, parameter->d_SXxz, parameter->d_SYxy, parameter->d_SYyy, parameter->d_SYyz, parameter->d_SZxz, parameter->d_SZyz, parameter->d_SZzz, parameter->d_e_dxi, parameter->d_dxi, parameter->d_e_dxi2, parameter->d_dxi2, parameter->d_e_dyj, parameter->d_dyj, parameter->d_e_dyj2, parameter->d_dyj2, parameter->d_e_dzk, parameter->d_dzk, parameter->d_dzk2, parameter->d_e_dzk2, parameter->d_ss, parameter->d_vwx, parameter->d_vwy, parameter->d_vwz, parameter->d_vwx2, parameter->d_vwy2, parameter->d_vwz2, parameter->d_SXss, parameter->d_SYss, parameter->d_SZss, parameter->d_pmlxss, parameter->d_pmlyss, parameter->d_pmlzss, parameter->d_VelocityWParameter1x, parameter->d_VelocityWParameter1y, parameter->d_VelocityWParameter1z, parameter->d_VelocityWParameter2x, parameter->d_VelocityWParameter2y, parameter->d_VelocityWParameter2z, parameter->d_VelocityWParameter3x, parameter->d_VelocityWParameter3y, parameter->d_VelocityWParameter3z, parameter->d_VelocityUParameter1x, parameter->d_VelocityUParameter1y, parameter->d_VelocityUParameter1z, parameter->d_VelocityUParameter2x, parameter->d_VelocityUParameter2y, parameter->d_VelocityUParameter2z, DT, nxt, nyt, nzt, bottom);
	//recv Vmid NT
	Recv_halo_mid_V(myid, coord, parameter, req_halo_mid, req_halore_mid, &send_count_mid, &recv_count_mid, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
	//pack Vbottom NT
	Pack_haloVbottom(myid, bottom, coord, parameter, nxt, nyt, nzt, compute_stream);
	//send Vbottom NT
	Send_halo_bottom_V(myid, coord, parameter, req_halo_down, req_halore_down, &send_count_down, &recv_count_down, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);



	
	//wait Vmid NT
    Wait_halo_send(req_halo_mid, send_count_mid, req_halore_mid, recv_count_mid);
	Wait_halo_recv(req_halo_mid, send_count_mid, req_halore_mid, recv_count_mid);
	send_count_mid = 0;
	recv_count_mid = 0;
	//unpack Vmid NT
	Unpack_haloVmid(mid, coord, parameter, nxt, nyt, nzt, compute_stream);
	
	cudaDeviceSynchronize();
	//computer Tmid
	FD_Tmid<<<Gridsize, Blocksize, 0, compute_stream>>>(parameter->d_vux, parameter->d_vuy, parameter->d_vuz, parameter->d_txx, parameter->d_tzz, parameter->d_tyy, parameter->d_txz, parameter->d_txy, parameter->d_tyz, parameter->d_pmlxVux, parameter->d_pmlyVuy, parameter->d_pmlzVuz, parameter->d_pmlxVuy, parameter->d_pmlyVux, parameter->d_pmlyVuz, parameter->d_pmlzVuy, parameter->d_pmlzVux, parameter->d_pmlxVuz, parameter->d_Vuxx, parameter->d_Vuxy, parameter->d_Vuxz, parameter->d_Vuyx, parameter->d_Vuyy, parameter->d_Vuyz, parameter->d_Vuzx, parameter->d_Vuzy, parameter->d_Vuzz, parameter->d_e_dxi, parameter->d_dxi, parameter->d_e_dxi2, parameter->d_dxi2, parameter->d_e_dyj2, parameter->d_dyj2, parameter->d_e_dyj, parameter->d_dyj, parameter->d_dzk2, parameter->d_e_dzk2, parameter->d_e_dzk, parameter->d_dzk, parameter->d_Vwxx, parameter->d_Vwyy, parameter->d_Vwzz, parameter->d_vwx, parameter->d_vwy, parameter->d_vwz, parameter->d_ss, parameter->d_pmlxVwx, parameter->d_pmlyVwy, parameter->d_pmlzVwz, parameter->d_PressParameter1, parameter->d_PressParameter2, parameter->d_StressParameter1, parameter->d_StressParameter2, parameter->d_StressParameter3, parameter->d_StressParameterxy, parameter->d_StressParameterxz, parameter->d_StressParameteryz, DT, nxt, nyt, nzt, mid);
	//recv Vtop NT
	Recv_halo_top_V(myid, coord, parameter, req_halo_up, req_halore_up, &send_count_up, &recv_count_up, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
    //recv Vbottom NT
	Recv_halo_bottom_V(myid, coord, parameter, req_halo_down, req_halore_down, &send_count_down, &recv_count_down, up, down, back, forward, left, right, MCW, nxt, nyt, nzt);
	

	
	// wait Vtop NT
    Wait_halo_recv(req_halo_up, send_count_up, req_halore_up, recv_count_up);
	Wait_halo_send(req_halo_up, send_count_up, req_halore_up, recv_count_up);
	send_count_up = 0;
	recv_count_up = 0;
	//unpack Vtop NT
    Unpack_haloVtop(top, coord, parameter, nxt, nyt, nzt, compute_stream);
	
	cudaDeviceSynchronize();
	//computer Ttop NT
	FD_Ttop<<<Gridsize, Blocksize, 0, compute_stream>>>(parameter->d_vux, parameter->d_vuy, parameter->d_vuz, parameter->d_txx, parameter->d_tzz, parameter->d_tyy, parameter->d_txz, parameter->d_txy, parameter->d_tyz, parameter->d_pmlxVux, parameter->d_pmlyVuy, parameter->d_pmlzVuz, parameter->d_pmlxVuy, parameter->d_pmlyVux, parameter->d_pmlyVuz, parameter->d_pmlzVuy, parameter->d_pmlzVux, parameter->d_pmlxVuz, parameter->d_Vuxx, parameter->d_Vuxy, parameter->d_Vuxz, parameter->d_Vuyx, parameter->d_Vuyy, parameter->d_Vuyz, parameter->d_Vuzx, parameter->d_Vuzy, parameter->d_Vuzz, parameter->d_e_dxi, parameter->d_dxi, parameter->d_e_dxi2, parameter->d_dxi2, parameter->d_e_dyj2, parameter->d_dyj2, parameter->d_e_dyj, parameter->d_dyj, parameter->d_dzk2, parameter->d_e_dzk2, parameter->d_e_dzk, parameter->d_dzk, parameter->d_Vwxx, parameter->d_Vwyy, parameter->d_Vwzz, parameter->d_vwx, parameter->d_vwy, parameter->d_vwz, parameter->d_ss, parameter->d_pmlxVwx, parameter->d_pmlyVwy, parameter->d_pmlzVwz, parameter->d_PressParameter1, parameter->d_PressParameter2, parameter->d_StressParameter1, parameter->d_StressParameter2, parameter->d_StressParameter3, parameter->d_StressParameterxy, parameter->d_StressParameterxz, parameter->d_StressParameteryz, DT, nxt, nyt, nzt, top);
	

	
	//wait Vbottom NT
	Wait_halo_send(req_halo_down, send_count_down, req_halore_down, recv_count_down);
	Wait_halo_recv(req_halo_down, send_count_down, req_halore_down, recv_count_down);
	send_count_down = 0;
	recv_count_down = 0;
	//unpack Vbotom NT
	Unpack_haloVbottom(bottom, coord, parameter, nxt, nyt, nzt, compute_stream);
	
	cudaDeviceSynchronize();
	//computer Tbottom
	FD_Tbottom<<<Gridsize, Blocksize, 0, compute_stream>>>(parameter->d_vux, parameter->d_vuy, parameter->d_vuz, parameter->d_txx, parameter->d_tzz, parameter->d_tyy, parameter->d_txz, parameter->d_txy, parameter->d_tyz, parameter->d_pmlxVux, parameter->d_pmlyVuy, parameter->d_pmlzVuz, parameter->d_pmlxVuy, parameter->d_pmlyVux, parameter->d_pmlyVuz, parameter->d_pmlzVuy, parameter->d_pmlzVux, parameter->d_pmlxVuz, parameter->d_Vuxx, parameter->d_Vuxy, parameter->d_Vuxz, parameter->d_Vuyx, parameter->d_Vuyy, parameter->d_Vuyz, parameter->d_Vuzx, parameter->d_Vuzy, parameter->d_Vuzz, parameter->d_e_dxi, parameter->d_dxi, parameter->d_e_dxi2, parameter->d_dxi2, parameter->d_e_dyj2, parameter->d_dyj2, parameter->d_e_dyj, parameter->d_dyj, parameter->d_dzk2, parameter->d_e_dzk2, parameter->d_e_dzk, parameter->d_dzk, parameter->d_Vwxx, parameter->d_Vwyy, parameter->d_Vwzz, parameter->d_vwx, parameter->d_vwy, parameter->d_vwz, parameter->d_ss, parameter->d_pmlxVwx, parameter->d_pmlyVwy, parameter->d_pmlzVwz, parameter->d_PressParameter1, parameter->d_PressParameter2, parameter->d_StressParameter1, parameter->d_StressParameter2, parameter->d_StressParameter3, parameter->d_StressParameterxy, parameter->d_StressParameterxz, parameter->d_StressParameteryz, DT, nxt, nyt, nzt, bottom);

	

	//output
	CUDA_CHECK(cudaMemcpy(h_txx, parameter->d_txx, mem_sizeHalf, cudaMemcpyDeviceToHost));
	//MPI_Barrier(MCW);
	for (iz = 0; iz < nzt_pro; iz++)
	{
		for (iy = 0; iy < nyt_pro; iy++)
		{
			for (ix = 0; ix < nxt_pro; ix++)
			{
				txx3d[iz][iy][ix] = h_txx[iz * nxt_pro * nyt_pro + iy * nxt_pro + ix];
			}
		}
	}
	if (coord[0] == 1 && coord[1] == 1 && coord[2] == 0)
	{
		if (it % 1 == 0)
		{
			for (int sisz = 0; sisz < nzt_pro; sisz++)
			{

				sis_x[sisz][it] = txx3d[sisz][synew][sxnew];
			}
		}
	}
	if (coord[0] == 1 && coord[1] == 1 && coord[2] == 1)
	{
		if (it % 1 == 0)
		{
			for (int sisz = 0; sisz < nzt_pro; sisz++)
			{
				sis_x2[sisz][it] = txx3d[sisz][synew][sxnew];
			}
		}
	}
	if (coord[0] == 1 && coord[1] == 1 && coord[2] == 2)
	{
		if (it % 1 == 0)
		{
			for (int sisz = 0; sisz < nzt_pro; sisz++)
			{
				sis_x3[sisz][it] = txx3d[sisz][synew][sxnew];
			}
		}
	}
	//------------------------------------------------------------------------------------------------------------------
	if (coord[0] == 1 && coord[1] == 1 && coord[2] == 0)
	{
		char sisxname[] = "sisx.dat";
		wfile(sisxname, sis_x, nzt_pro, NT);
	}
	if (coord[0] == 1 && coord[1] == 1 && coord[2] == 1)
	{
		char sisx2name[] = "sisx2.dat";
		wfile(sisx2name, sis_x2, nzt_pro, NT);
	}
	if (coord[0] == 1 && coord[1] == 1 && coord[2] == 2)
	{
		char sisx3name[] = "sisx3.dat";
		wfile(sisx3name, sis_x3, nzt_pro, NT);
	}
	//free
	cudaFree(parameter->halo_Vuz_up_pack);
	cudaFree(parameter->halo_Vwz_up_pack);
	cudaFree(parameter->halo_Txz_up_pack);
	cudaFree(parameter->halo_Tyz_up_pack);
	cudaFree(parameter->halo_Vuz_up_recv);
	cudaFree(parameter->halo_Vwz_up_recv);
	cudaFree(parameter->halo_Txz_up_recv);
	cudaFree(parameter->halo_Tyz_up_recv);
	cudaFree(parameter->halo_Vux_down_pack);
	cudaFree(parameter->halo_Vuy_down_pack);
	cudaFree(parameter->halo_Tzz_down_pack);
	cudaFree(parameter->halo_ss_down_pack);
	cudaFree(parameter->halo_Vux_down_recv);
	cudaFree(parameter->halo_Vuy_down_recv);
	cudaFree(parameter->halo_Tzz_down_recv);
	cudaFree(parameter->halo_ss_down_recv);
	//top
	cudaFree(parameter->halo_Vux_right_top_pack);
	cudaFree(parameter->halo_Vux_back_top_pack);
	cudaFree(parameter->halo_Vuy_left_top_pack);
	cudaFree(parameter->halo_Vuy_front_top_pack);
	cudaFree(parameter->halo_Vuz_left_top_pack);
	cudaFree(parameter->halo_Vuz_back_top_pack);
	cudaFree(parameter->halo_Vwx_right_top_pack);
	cudaFree(parameter->halo_Vwy_front_top_pack);
	cudaFree(parameter->halo_Txx_left_top_pack);
	cudaFree(parameter->halo_Txy_right_top_pack);
	cudaFree(parameter->halo_Txy_front_top_pack);
	cudaFree(parameter->halo_Txz_right_top_pack);
	cudaFree(parameter->halo_Tyy_back_top_pack);
	cudaFree(parameter->halo_Tyz_front_top_pack);
  	cudaFree(parameter->halo_ss_left_top_pack);
	cudaFree(parameter->halo_ss_back_top_pack);
	cudaFree(parameter->halo_Vux_right_top_recv);
	cudaFree(parameter->halo_Vux_back_top_recv);
	cudaFree(parameter->halo_Vuy_left_top_recv);
	cudaFree(parameter->halo_Vuy_front_top_recv);
	cudaFree(parameter->halo_Vuz_left_top_recv);
	cudaFree(parameter->halo_Vuz_back_top_recv);
	cudaFree(parameter->halo_Vwx_right_top_recv);
	cudaFree(parameter->halo_Vwy_front_top_recv);
	cudaFree(parameter->halo_Txx_left_top_recv);
	cudaFree(parameter->halo_Txy_right_top_recv);
	cudaFree(parameter->halo_Txy_front_top_recv);
	cudaFree(parameter->halo_Txz_right_top_recv);
	cudaFree(parameter->halo_Tyy_back_top_recv);
	cudaFree(parameter->halo_Tyz_front_top_recv);
  	cudaFree(parameter->halo_ss_left_top_recv);
	cudaFree(parameter->halo_ss_back_top_recv);
	//mid
	cudaFree(parameter->halo_Vux_right_mid_pack);
	cudaFree(parameter->halo_Vux_back_mid_pack);
	cudaFree(parameter->halo_Vuy_left_mid_pack);
	cudaFree(parameter->halo_Vuy_front_mid_pack);
	cudaFree(parameter->halo_Vuz_left_mid_pack);
	cudaFree(parameter->halo_Vuz_back_mid_pack);
	cudaFree(parameter->halo_Vwx_right_mid_pack);
	cudaFree(parameter->halo_Vwy_front_mid_pack);
	cudaFree(parameter->halo_Txx_left_mid_pack);
	cudaFree(parameter->halo_Txy_right_mid_pack);
	cudaFree(parameter->halo_Txy_front_mid_pack);
	cudaFree(parameter->halo_Txz_right_mid_pack);
	cudaFree(parameter->halo_Tyy_back_mid_pack);
	cudaFree(parameter->halo_Tyz_front_mid_pack);
  	cudaFree(parameter->halo_ss_left_mid_pack);
	cudaFree(parameter->halo_ss_back_mid_pack);
	cudaFree(parameter->halo_Vux_right_mid_recv);
	cudaFree(parameter->halo_Vux_back_mid_recv);
	cudaFree(parameter->halo_Vuy_left_mid_recv);
	cudaFree(parameter->halo_Vuy_front_mid_recv);
	cudaFree(parameter->halo_Vuz_left_mid_recv);
	cudaFree(parameter->halo_Vuz_back_mid_recv);
	cudaFree(parameter->halo_Vwx_right_mid_recv);
	cudaFree(parameter->halo_Vwy_front_mid_recv);
	cudaFree(parameter->halo_Txx_left_mid_recv);
	cudaFree(parameter->halo_Txy_right_mid_recv);
	cudaFree(parameter->halo_Txy_front_mid_recv);
	cudaFree(parameter->halo_Txz_right_mid_recv);
	cudaFree(parameter->halo_Tyy_back_mid_recv);
	cudaFree(parameter->halo_Tyz_front_mid_recv);
  	cudaFree(parameter->halo_ss_left_mid_recv);
	cudaFree(parameter->halo_ss_back_mid_recv);
	//bottom
	cudaFree(parameter->halo_Vux_right_bottom_pack);
	cudaFree(parameter->halo_Vux_back_bottom_pack);
	cudaFree(parameter->halo_Vuy_left_bottom_pack);
	cudaFree(parameter->halo_Vuy_front_bottom_pack);
	cudaFree(parameter->halo_Vuz_left_bottom_pack);
	cudaFree(parameter->halo_Vuz_back_bottom_pack);
	cudaFree(parameter->halo_Vwx_right_bottom_pack);
	cudaFree(parameter->halo_Vwy_front_bottom_pack);
	cudaFree(parameter->halo_Txx_left_bottom_pack);
	cudaFree(parameter->halo_Txy_right_bottom_pack);
	cudaFree(parameter->halo_Txy_front_bottom_pack);
	cudaFree(parameter->halo_Txz_right_bottom_pack);
	cudaFree(parameter->halo_Tyy_back_bottom_pack);
	cudaFree(parameter->halo_Tyz_front_bottom_pack);
  	cudaFree(parameter->halo_ss_left_bottom_pack);
	cudaFree(parameter->halo_ss_back_bottom_pack);
	cudaFree(parameter->halo_Vux_right_bottom_recv);
	cudaFree(parameter->halo_Vux_back_bottom_recv);
	cudaFree(parameter->halo_Vuy_left_bottom_recv);
	cudaFree(parameter->halo_Vuy_front_bottom_recv);
	cudaFree(parameter->halo_Vuz_left_bottom_recv);
	cudaFree(parameter->halo_Vuz_back_bottom_recv);
	cudaFree(parameter->halo_Vwx_right_bottom_recv);
	cudaFree(parameter->halo_Vwy_front_bottom_recv);
	cudaFree(parameter->halo_Txx_left_bottom_recv);
	cudaFree(parameter->halo_Txy_right_bottom_recv);
	cudaFree(parameter->halo_Txy_front_bottom_recv);
	cudaFree(parameter->halo_Txz_right_bottom_recv);
	cudaFree(parameter->halo_Tyy_back_bottom_recv);
	cudaFree(parameter->halo_Tyz_front_bottom_recv);
  	cudaFree(parameter->halo_ss_left_bottom_recv);
	cudaFree(parameter->halo_ss_back_bottom_recv);
	free_space2d(sis_x, nzt_pro);
	free_space2d(sis_x2, nzt_pro);
	free_space2d(sis_x3, nzt_pro);
	free_space3d(txx3d, nzt_pro, nyt_pro);
	free_space3d(StressParameterxz, nzt_pro, nyt_pro);
	free_space3d(ddzk, nzt_pro, nyt_pro);
	free(h_txx);
	free(h_StressParameterxz);
	free(h_dzk);
}
//*************************************************************************************************************************
//*************************************************************************************************************************