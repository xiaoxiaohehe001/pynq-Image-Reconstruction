#include <ap_int.h>
#include <ap_fixed.h>
#include <math.h>

typedef float Dtype_f;
typedef float Dtype_w;
typedef float Dtype_mul;
typedef float Dtype_acc;
using namespace std;

void Conv2d(
		ap_uint<16> CHin, ap_uint<16> CHout,
		ap_uint<16> Hin,  ap_uint<16> Win,
		ap_uint<8> kx,   ap_uint<8> ky,
		ap_uint<8> sx,   ap_uint<8> sy,
		ap_uint<1> mode, ap_uint<1> relu_en,
		Dtype_f feature_in[], Dtype_w W[],
		Dtype_w bias[],  Dtype_f feature_out[]
)   //mode参数值固定为0，即全部为0,relu_en为0，执行InstanceNorm之后再执行relu_en
{
	//规定形参类型
	#pragma HLS INTERFACE m_axi depth=4294967295 port=feature_out offset=slave
	#pragma HLS INTERFACE m_axi depth=4294967295 port=bias offset=slave
	#pragma HLS INTERFACE m_axi depth=4294967295 port=W offset=slave
	#pragma HLS INTERFACE m_axi depth=4294967295 port=feature_in offset=slave
	#pragma HLS INTERFACE s_axilite port=relu_en
	#pragma HLS INTERFACE s_axilite port=CHout
	#pragma HLS INTERFACE s_axilite port=sx
	#pragma HLS INTERFACE s_axilite port=Hin
	#pragma HLS INTERFACE s_axilite port=CHin
	#pragma HLS INTERFACE s_axilite port=kx
	#pragma HLS INTERFACE s_axilite port=mode
	#pragma HLS INTERFACE s_axilite port=sy
	#pragma HLS INTERFACE s_axilite port=ky
	#pragma HLS INTERFACE s_axilite port=Win
	#pragma HLS INTERFACE s_axilite port=return

	ap_uint<8> pad_x,pad_y;
	if(mode==0)
	{
		pad_x=0;pad_y=0;
	}
	else
	{
		pad_x=(kx-1)/2;pad_y=(ky-1)/2;
	}
	ap_uint<16> Hout,Wout;
	Wout=(Win+2*pad_x-kx)/sx+1;
	Hout=(Hin+2*pad_y-ky)/sy+1;

	for(int cout=0;cout<CHout;cout++)
		for(int i=0;i<Hout;i++)
			for(int j=0;j<Wout;j++)
			{
				Dtype_acc sum=0;
				for(int ii=0;ii<ky;ii++)
					for(int jj=0;jj<kx;jj++)
					{
						ap_int<16> h=i*sy-pad_y+ii;
						ap_int<16> w=j*sx-pad_x+jj;
						if(h>=0 && w>=0 && h<Hin && w<Win)
						{
							for(int cin=0;cin<CHin;cin++)
							{
								Dtype_mul tp=feature_in[h*CHin*Win+w*CHin+cin]*W[ii*kx*CHin*CHout+jj*CHin*CHout+cin*CHout+cout];
								sum+=tp;
							}
						}
					}

				sum+=bias[cout];
				if(relu_en && sum<0)
					sum=0;
				feature_out[i*Wout*CHout+j*CHout+cout]=sum;
			}
}
