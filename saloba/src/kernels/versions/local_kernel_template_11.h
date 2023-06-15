#ifndef __LOCAL_KERNEL_TEMPLATE__
#define __LOCAL_KERNEL_TEMPLATE__


// This old core provides the same result as the currently LOCAL core, but lacks some optimization. Left for historical / comparative purposes.
#define CORE_LOCAL_DEPRECATED_COMPUTE() \
		uint32_t gbase = (gpac >> l) & 15;/*get a base from target_batch sequence */ \
		DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase);/* check equality of rbase and gbase */ \
		f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);/* whether to introduce or extend a gap in query_batch sequence */ \
		h[m] = p[m] + subScore; /*score if rbase is aligned to gbase*/ \
		h[m] = max(h[m], f[m]); \
		h[m] = max(h[m], 0); \
		e = max(h[m - 1] - _cudaGapOE, e - _cudaGapExtend);/*whether to introduce or extend a gap in target_batch sequence */\
		h[m] = max(h[m], e); \
		maxXY_y = (maxHH < h[m]) ? gidx + (m-1) : maxXY_y; \
		maxHH = (maxHH < h[m]) ? h[m] : maxHH; \
		p[m] = h[m-1];

#define CORE_LOCAL_COMPUTE() \
		uint32_t gbase = (gpac >> l) & 15;\
		DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase) \
		int32_t tmp_hm = p[m] ? p[m] + subScore : 0; \
		h[m] = max(tmp_hm, f[m]); \
		h[m] = max(h[m], e); \
		h[m] = max(h[m], 0); \
		f[m] = max(tmp_hm- _cudaGapOE, f[m] - _cudaGapExtend); \
		e = max(tmp_hm- _cudaGapOE, e - _cudaGapExtend); \
		p[m] = h[m-1]; \

#define CORE_LOCAL_COMPUTE_START() \
		uint32_t gbase = (gpac >> l) & 15;\
		DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase) \
		int32_t tmp_hm = p[m] + subScore; \
		h[m] = max(tmp_hm, f[m]); \
		h[m] = max(h[m], e); \
		h[m] = max(h[m], 0); \
		f[m] = max(tmp_hm- _cudaGapOE, f[m] - _cudaGapExtend); \
		e = max(tmp_hm- _cudaGapOE, e - _cudaGapExtend); \
		maxXY_y = (maxHH < h[m]) ? gidx + (m-1) : maxXY_y; \
		maxHH = (maxHH < h[m]) ? h[m] : maxHH; \
		p[m] = h[m-1]; \

#define CORE_LOCAL_COMPUTE_TB(direction_reg) \
		uint32_t gbase = (gpac >> l) & 15;\
		DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase) \
		int32_t tmp_hm = p[m] + subScore; \
		uint32_t m_or_x = tmp_hm >= p[m] ? 0 : 1;\
		h[m] = max(tmp_hm, f[m]); \
		h[m] = max(h[m], e); \
		h[m] = max(h[m], 0); \
		direction_reg |= h[m] == tmp_hm ? m_or_x << (28 - ((m - 1) << 2)) : (h[m] == f[m] ? (uint32_t)3 << (28 - ((m - 1) << 2)) : (uint32_t)2 << (28 - ((m - 1) << 2)));\
		direction_reg |= (tmp_hm - _cudaGapOE) > (f[m] - _cudaGapExtend) ?  (uint32_t)0 : (uint32_t)1 << (31 - ((m - 1) << 2));\
		f[m] = max(tmp_hm- _cudaGapOE, f[m] - _cudaGapExtend); \
		direction_reg |= (tmp_hm - _cudaGapOE) > (e - _cudaGapExtend) ?  (uint32_t)0 : (uint32_t)1 << (30 - ((m - 1) << 2));\
		e = max(tmp_hm- _cudaGapOE, e - _cudaGapExtend); \
		maxXY_y = (maxHH < h[m]) ? gidx + (m-1) : maxXY_y; \
		maxHH = (maxHH < h[m]) ? h[m] : maxHH; \
		p[m] = h[m-1]; \




/* typename meaning : 
    - T is the algorithm type (LOCAL, MICROLOCAL)
    - S is WITH_ or WIHTOUT_START
    - B is for computing the Second Best Score. Its values are on enum FALSE(0)/TRUE(1).
    (sidenote: it's based on an enum instead of a bool in order to generalize its type from its Int value, with Int2Type meta-programming-template)
*/

enum my{
	threads_per_pair = 16,
	threadIdx_y = 4,
	seqs_per_block = 32 * threadIdx_y / threads_per_pair,
};

template <typename T, typename S, typename B>
__global__ void gasal_local_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, gasal_res_t *device_res, int n_tasks, short2 *device_bottom, int *dev_h0, int band_width, int z_drop)
{
	int16_t seqid = blockIdx.x * blockDim.y * (blockDim.x / threads_per_pair) + threadIdx.y * (blockDim.x / threads_per_pair) + threadIdx.x / threads_per_pair;
	int16_t tid = threadIdx.x % threads_per_pair;
	if(seqid >= n_tasks) return;
	int16_t seqId_block = threadIdx.y * blockDim.x / threads_per_pair + threadIdx.x / threads_per_pair;

	//--------------------------------------------------------------------------------------------------------------------------
	uint32_t packed_target_batch_idx = target_batch_offsets[seqid] >> 3; //starting index of the target_batch sequence
	uint32_t packed_query_batch_idx = query_batch_offsets[seqid] >> 3;//starting index of the query_batch sequence
	uint32_t read_len = query_batch_lens[seqid];
	uint32_t ref_len = target_batch_lens[seqid];
	uint32_t query_batch_regs = (read_len >> 3) + (read_len&7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
	uint32_t target_batch_regs = (ref_len >> 3) + (ref_len&7 ? 1 : 0);//number of 32-bit words holding target_batch sequence
	//--------------------------------------------------------------------------------------------------------------------------
	//flag. should initialize device_bottom?
	short2 HD; //used for one cell.
	int16_t h[9];
	int16_t f[9];
	int16_t p[9];
	
	uint16_t i;
	int16_t k, m, l; //[remove i->m]
	int16_t e;
	int16_t subScore;
	//----------------------------------------
	uint16_t full_chunk_num = target_batch_regs / threads_per_pair; //[remove]
	uint16_t last_chunk_height = target_batch_regs % threads_per_pair; //[remove]
	uint16_t chunk_idx; 
	uint16_t step;
	uint32_t col_block;
	uint32_t col_cell;

	int16_t step_q;	//[remove]
	int16_t step_r;	//[remove]
	int16_t offset_partial_bottom; //[remove]
	int16_t offset_device_bottom; //[remove]

	//const int band_width = w;//[banded]
	int16_t global_row = 0;
	int16_t global_col = 0;
	int16_t band_start = 0;
	int16_t band_end = 0;
	int16_t local_start = 0;
	int16_t local_end = 0;

	int16_t h0 = dev_h0[seqid];
	bool is_end_last = 0;


	__shared__ short2 partial_bottom[seqs_per_block][threads_per_pair * 2 * 8];
	__shared__ int row_max[seqs_per_block][threads_per_pair * 8];
	__shared__ int row_max_col[seqs_per_block][threads_per_pair * 8];
	__shared__ short4 thread_max[seqs_per_block][threads_per_pair + 1]; 
	__shared__ int thread_max_off[seqs_per_block][threads_per_pair + 1];
	__shared__ short2 thread_g[seqs_per_block][threads_per_pair + 1];
	__shared__ uint32_t bases[seqs_per_block][threads_per_pair * 2];
	//----------------------------------------
	//initialize-----------------------------
	//[can be optimized. we know h0, then means know number of interatino]
	short2 initHD = make_short2(0, 0);
	device_bottom = &(device_bottom[seqid * MAX_QUERY_LEN]);
	
	for(i = 0; i < (int)(read_len / threads_per_pair) + 2; ++i){
		int h = h0 - _cudaGapOE - _cudaGapExtend * (i*threads_per_pair + tid);
		h = h > 0 ? h : 0;
		initHD.x = h;
		device_bottom[i * threads_per_pair + tid] = initHD; //[i*threads_per_pair + tid]
	}
	short4 SRCE;
	thread_max[seqId_block][tid] = make_short4(0, 0, 0, threads_per_pair);
	if(tid ==0){
		thread_max[seqId_block][0].x = h0;
		thread_max[seqId_block][0].y = -1;
		thread_max[seqId_block][0].z = -1;
		thread_max[seqId_block][0].w = threads_per_pair;
	}
	thread_max_off[seqId_block][tid] = 0;
	thread_max_off[seqId_block][tid+1] = 0;
	initHD.x = -1; initHD.y = -1;
	if(tid == 0) thread_g[seqId_block][tid] = initHD;
	thread_g[seqId_block][tid+1] = initHD;
	__syncwarp();
	//---------------------------------------
	//for full chunks
	for(chunk_idx = 0; chunk_idx < full_chunk_num; ++chunk_idx){
		col_block = 0;
		is_end_last = 0;
		for (m = 0; m < 9; m++) {
            h[m] = 0;
            f[m] = 0;
            p[m] = 0;
			//row_max[m] = 0;
			//row_max_col[seqId_block][tid * 8 + m - 1] = 0;
        }
		for(m = 1; m < 9; ++m){
			row_max[seqId_block][tid * 8 + m - 1] = 0;
			row_max_col[seqId_block][tid * 8 + m - 1] = 0;
		}
		//----------------------------
		for(int c = 0; c < 8; ++c){
			partial_bottom[seqId_block][tid*8 + c] = device_bottom[tid*8 + c];
			partial_bottom[seqId_block][(tid + threads_per_pair)*8 + c] = device_bottom[(tid + threads_per_pair)*8 + c];
		}
		//-------------------------	
		global_row = 8 * threads_per_pair * chunk_idx + 8 * tid;
		global_col = 0;
		band_start = (global_col) - band_width; 
		band_end = (global_col) + band_width;
		//-------------------------
		
		//-------------------------
		if(!(band_end < global_row || global_row + 7 < band_start)) {
			local_start = band_start - global_row;
			local_end = band_end - global_row;
			if(local_start < 0) local_start = 0;
			if(local_end > 7) local_end = 7;
			//global_row + local_start <= =< global_row + local_end
			for(i = local_start; i < local_end + 1; ++i){
				p[i+1] = h0 - _cudaGapOE - _cudaGapExtend*(global_row + i - 1);
				p[i+1] = p[i+1] > 0 ? p[i+1] : 0;
			}
		}
		if(chunk_idx == 0 && tid == 0){
			p[1] = h0;
		}
		//------------------------
		bases[seqId_block][tid] = packed_query_batch[packed_query_batch_idx + tid];
		//bases[seqId_block][threads_per_pair + tid] = packed_query_batch[packed_query_batch_idx + threads_per_pair + tid];
		__syncwarp();
		//-------------------------
		register uint32_t gpac =packed_target_batch[packed_target_batch_idx + threads_per_pair * chunk_idx + tid];//load 8 packed bases from target_batch sequence
		for(step = 0; step < threads_per_pair + query_batch_regs -1;){
			if(tid <= step && step < tid + query_batch_regs /*&& band_start <= global_row + 7*/){
				//-------------initialize-------------------------
				//if(band_end + 7 < global_row || band_start > global_row + 7) continue;
				//read ref and qeury regs
				register uint32_t rpac =bases[seqId_block][col_block % (2*threads_per_pair)];//load 8 bases from query_batch sequence
				//fill 8x8cell
				col_cell = 0;
				for (k = 28; k >= 0; k -= 4, ++col_cell, ++global_col, ++band_start, ++band_end) {
					//--------------decide row range--------------
					if(band_end < global_row || global_row + 7 < band_start) {
						p[1] = partial_bottom[seqId_block][(col_block % (threads_per_pair * 2))*8 + col_cell].x;
						continue;
					}
					local_start = (band_start - global_row);
					local_end = (band_end - global_row);
					if(local_start < 0) local_start = 0;
					if(local_end > 7) local_end = 7;
					//local_start = 0;
					//local_end = 7;
					//--------------------------------------------
					uint32_t rbase = (rpac >> k) & 15;//get a base from query_batch sequence
					//-----load intermediate values--------------
					//read bottom cell of upper block from partial_bottom[threadIdx.y][col]
					HD = partial_bottom[seqId_block][(col_block % (threads_per_pair * 2))*8 + col_cell];
					h[0] = HD.x;
					e = HD.y;
					//--------------------------------------------
//#pragma unroll 8
					for (l = 28 - 4*local_start, m = 1 + local_start; m < 1 + (local_end + 1); l -= 4, m++) {
						CORE_LOCAL_COMPUTE();
						row_max[seqId_block][tid * 8 + m - 1] = row_max[seqId_block][tid * 8 + m - 1] > h[m] ? row_max[seqId_block][tid * 8 + m - 1] : h[m];
						row_max_col[seqId_block][tid * 8 + m - 1] = row_max[seqId_block][tid * 8 + m - 1] > h[m] ? row_max_col[seqId_block][tid * 8 + m - 1] : global_col;
					}
					p[m] = h[m-1];
					//----------save intermediate values------------
					//save result on shared_memory[~][col]
					HD.x = h[m-1];
					HD.y = e;
					partial_bottom[seqId_block][(col_block % (threads_per_pair * 2))*8 + col_cell] = HD;

					//-------------------------------------------------------
					if(global_col == read_len - 1){
						is_end_last = 1;
						break;
					}
					//-------------------------------------------------------
				}
				++col_block;
			}
			__syncwarp();
			//save partial_bottom onto global memory
			++step;
			step_q = (step) / threads_per_pair;
			step_r = (step) % threads_per_pair;
			if(step_q >= 2 && step_r == 0){
				offset_partial_bottom = (step_q % 2) * threads_per_pair;
				offset_device_bottom = (step_q -2) * threads_per_pair;
				offset_device_bottom = (offset_device_bottom + tid) * 8;
				for(i = 0; i < 8; ++i){
					device_bottom[offset_device_bottom + i] = partial_bottom[seqId_block][(offset_partial_bottom + tid)*8 + i];
				}
				//---------------------------------
				for(i = 0; i < 8; ++i){
					partial_bottom[seqId_block][(offset_partial_bottom + tid)*8 + i] = device_bottom[offset_device_bottom + threads_per_pair*2*8 + i];
				}
				//--------------------------------
			}
			if(step % (threads_per_pair) == 0){
				//((step/threads_per_pair)/2)
				bases[seqId_block][((step/threads_per_pair)%2)*threads_per_pair + tid] = packed_query_batch[packed_query_batch_idx + step + tid];
			}
			__syncwarp();

		}
		//after all steps, save residues of partial bottom to global
		//++step;//{????????????????????????}
		step_q = (step) / threads_per_pair;
		step_r = (step) % threads_per_pair;
		//if(step_r != 0){
			++step_q;
			offset_partial_bottom = (step_q % 2) * threads_per_pair;
			offset_device_bottom = (step_q -2) * threads_per_pair;
			offset_device_bottom = (offset_device_bottom + tid) * 8;
			for(i = 0; i < 8; ++i){
				device_bottom[offset_device_bottom + i] = partial_bottom[seqId_block][(offset_partial_bottom + tid)*8 + i];
			}
			for(i = 0; i < 8; ++i){
				partial_bottom[seqId_block][(offset_partial_bottom + tid)*8 + i] = device_bottom[offset_device_bottom + threads_per_pair*2*8 + i];
			}
		//}
		__syncwarp();

		for(m = 1; m < 9; ++m){
			if(row_max[seqId_block][tid * 8 + m - 1] > thread_max[seqId_block][tid+1].x){
				thread_max[seqId_block][tid+1].x = row_max[seqId_block][tid * 8 + m - 1];
				thread_max[seqId_block][tid+1].y = global_row + m -1;
				thread_max[seqId_block][tid+1].z = row_max_col[seqId_block][tid * 8 + m - 1];
			}
		}

		//kogge-stone algorithm
		for(i = 1; i < threads_per_pair; i = i * 2){
			if(i <= tid && (thread_max[seqId_block][tid].x <= thread_max[seqId_block][tid-i].x)){
				SRCE.x = thread_max[seqId_block][tid-i].x;
				SRCE.y = thread_max[seqId_block][tid-i].y;
				SRCE.z = thread_max[seqId_block][tid-i].z;
				__syncwarp();
				thread_max[seqId_block][tid].x = SRCE.x;
				thread_max[seqId_block][tid].y = SRCE.y;
				thread_max[seqId_block][tid].z = SRCE.z;
			}
			else{
				__syncwarp();	
			}
			__syncwarp();
		}

		//check_early stop && z_drop && get accum max && gscore
		for(m = 1; m < 9; ++m){
			//early
			if(local_start <= (m-1) && (m-1) <= local_end && is_end_last){
				if(h[m] >= thread_g[seqId_block][tid+1].x && (global_row+m-1)<ref_len){
					thread_g[seqId_block][tid+1].x = h[m];
					thread_g[seqId_block][tid+1].y = global_row + m -1;
				}
			}
			if(row_max[seqId_block][tid * 8 + m - 1] == 0){
				thread_max[seqId_block][tid].w = tid;
				break;
			}
			if(row_max[seqId_block][tid * 8 + m - 1] > thread_max[seqId_block][tid].x){
				thread_max[seqId_block][tid].x = row_max[seqId_block][tid * 8 + m - 1];
				thread_max[seqId_block][tid].y = global_row + m - 1;
				thread_max[seqId_block][tid].z = row_max_col[seqId_block][tid * 8 + m - 1];

				int max_off = (global_row + m -1) - row_max_col[seqId_block][tid * 8 + m - 1];
				if(max_off < 0) max_off = -max_off;
				thread_max_off[seqId_block][tid+1] = thread_max_off[seqId_block][tid+1] > max_off ? thread_max_off[seqId_block][tid+1] : max_off;
			}
			//z-drop
			//thread_max[seqId_block][tid].y,z ~ (global_row+m-1)~row_max_col[seqId_block][tid * 8 + m - 1];
			if((thread_max[seqId_block][tid].y - (global_row + m -1)) > (thread_max[seqId_block][tid].z - row_max_col[seqId_block][tid * 8 + m - 1])){
				if(thread_max[seqId_block][tid].x - row_max[seqId_block][tid * 8 + m - 1] - ((thread_max[seqId_block][tid].y - (global_row + m -1)) - (thread_max[seqId_block][tid].z - row_max_col[seqId_block][tid * 8 + m - 1]))*_cudaGapExtend > z_drop){
					thread_max[seqId_block][tid].w = tid;
					break;
				}
			}
			else{
				if(thread_max[seqId_block][tid].x - row_max[seqId_block][tid * 8 + m - 1] - ((thread_max[seqId_block][tid].z - row_max_col[seqId_block][tid * 8 + m - 1]) - (thread_max[seqId_block][tid].y - (global_row + m -1)))*_cudaGapExtend > z_drop){
					thread_max[seqId_block][tid].w = tid;
					break;
				}
			}
		}

		__syncwarp();
		//again kogge-stone algorithm for finding final maximum
		for(i = 1; i < threads_per_pair; i = i * 2){
			int temp = 0;
			if(i <= tid){
				temp = thread_max[seqId_block][tid-i].w < thread_max[seqId_block][tid].w ? thread_max[seqId_block][tid-i].w : thread_max[seqId_block][tid].w;
				__syncwarp();
				thread_max[seqId_block][tid].w = temp;
			}
			else{
				__syncwarp();	
			}
			__syncwarp();
		}

		//kogge-stone algorithm for max_off. target idx is 1<= < threads_per_pair+1
		for(i = 1; i < threads_per_pair + 1; i = i * 2){
			int bigger = 0;
			if(i-1 <= tid){
				bigger = thread_max_off[seqId_block][tid + 1 -i] < thread_max_off[seqId_block][tid + 1 ] ? thread_max_off[seqId_block][tid + 1 ] : thread_max_off[seqId_block][tid + 1 -i];
				__syncwarp();
				thread_max_off[seqId_block][tid + 1 ] = bigger;
			}
			else{
				__syncwarp();	
			}
			__syncwarp();
		}

		//kogge-stone for gscore
		for(i = 1; i < threads_per_pair + 1; i = i * 2){
			short2 big = initHD;
			if(i-1 <= tid){
				big.x = thread_g[seqId_block][tid + 1 - i].x;
				big.y = thread_g[seqId_block][tid + 1 - i].y;
				if(thread_g[seqId_block][tid + 1].x >= thread_g[seqId_block][tid + 1 - i].x){
					big.x = thread_g[seqId_block][tid + 1].x;
					big.y = thread_g[seqId_block][tid + 1].y;
				}
				__syncwarp();
				thread_g[seqId_block][tid+1].x = big.x;
				thread_g[seqId_block][tid+1].y = big.y;
			}
			else{
				__syncwarp();	
			}
			__syncwarp();
		}
		//move max to zeroIdx for max and maxoff
		if(tid == 0){
			//max
			int idx = thread_max[seqId_block][threads_per_pair-1].w == threads_per_pair ? threads_per_pair -1 : thread_max[seqId_block][threads_per_pair-1].w;
			thread_max[seqId_block][0].x = thread_max[seqId_block][idx].x;
			thread_max[seqId_block][0].y = thread_max[seqId_block][idx].y;
			thread_max[seqId_block][0].z = thread_max[seqId_block][idx].z;
			thread_max[seqId_block][0].w = thread_max[seqId_block][idx].w;

			thread_max_off[seqId_block][0] = thread_max_off[seqId_block][idx + 1] > thread_max_off[seqId_block][0] ? thread_max_off[seqId_block][idx + 1] : thread_max_off[seqId_block][0];
			thread_g[seqId_block][0].y = thread_g[seqId_block][idx + 1].x >= thread_g[seqId_block][0].x ? thread_g[seqId_block][idx + 1].y : thread_g[seqId_block][0].y;
			thread_g[seqId_block][0].x = thread_g[seqId_block][idx + 1].x >= thread_g[seqId_block][0].x ? thread_g[seqId_block][idx + 1].x : thread_g[seqId_block][0].x; 
		}
		thread_max_off[seqId_block][tid+1] = 0;
		thread_g[seqId_block][tid+1] = initHD;
		__syncwarp();

		if(thread_max[seqId_block][0].w < threads_per_pair) break;

	}
		//////----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		//////---------------------------------------------------------------------------LAST CHUNK--------------------------------------------------------------------------------------
		//////----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

		for (m = 0; m < 9; m++) {
            h[m] = 0;
            f[m] = 0;
            p[m] = 0;
			//row_max[m] = 0;
			//row_max_col[seqId_block][tid * 8 + m - 1] = 0;
        }

		for( m = 1; m < 9; ++m){
			row_max[seqId_block][tid * 8 + m - 1] = 0;
			row_max_col[seqId_block][tid * 8 + m - 1] = 0;
		}

		for(int c = 0; c < 8; ++c){
			partial_bottom[seqId_block][tid*8 + c] = device_bottom[ tid*8 + c];
			partial_bottom[seqId_block][(tid + threads_per_pair)*8 + c] = device_bottom[(tid + threads_per_pair)*8 + c];
		}
		__syncwarp();
		col_block = 0;
		is_end_last = 0;
		//----------------------------------------------------
		global_row = 8 * threads_per_pair * chunk_idx + 8 * tid;
		global_col = 0;
		band_start = (global_col) - band_width; 
		band_end = (global_col) + band_width;
		//----------------------------------------------------
		
		//-------------------------
		if(!(band_end < global_row || global_row + 7 < band_start)) {
			local_start = band_start - global_row;
			local_end = band_end - global_row;
			if(local_start < 0) local_start = 0;
			if(local_end > 7) local_end = 7;
			for(i = local_start; i < local_end + 1; ++i){
				p[i+1] = h0 - _cudaGapOE - _cudaGapExtend*(global_row + i - 1);
				p[i+1] = p[i+1] > 0 ? p[i+1] : 0;
			}
		}
		if(chunk_idx == 0 && tid == 0){
			p[1] = h0;
		}
		//-------------------------
		bases[seqId_block][tid] = packed_query_batch[packed_query_batch_idx + tid];
		//bases[seqId_block][threads_per_pair + tid] = packed_query_batch[packed_query_batch_idx + threads_per_pair + tid];
		__syncwarp();
		register uint32_t gpac =packed_target_batch[packed_target_batch_idx + threads_per_pair * chunk_idx + tid];//load 8 packed bases from target_batch sequence
		for(step = 0; step < last_chunk_height + query_batch_regs -1;){
			//if(max_info[seqId_block][3]) break;
			if(thread_max[seqId_block][0].w < threads_per_pair) break;
			if(tid <= step && step < tid + query_batch_regs && tid < last_chunk_height /*&& band_start <= global_row + 7*/){
				//printf("last chunk\n");
				//read ref and qeury regs
				register uint32_t rpac =bases[seqId_block][col_block %(2*threads_per_pair)];//load 8 bases from query_batch sequence
				//fill 8x8cell
				col_cell = 0;
				for (k = 28; k >= 0; k -= 4, ++col_cell, ++global_col, ++band_start, ++band_end) {
					//--------------decide row range--------------
					if(band_end < global_row || global_row + 7 < band_start) {
						p[1] = partial_bottom[seqId_block][(col_block % (threads_per_pair * 2))*8 + col_cell].x;
						continue;
					}
					local_start = band_start - global_row;
					local_end = band_end - global_row;
					if(local_start < 0) local_start = 0;
					if(local_end > 7) local_end = 7;
					//--------------------------------------------
					uint32_t rbase = (rpac >> k) & 15;//get a base from query_batch sequence
					//-----load intermediate values--------------
					//read bottom cell of upper block from partial_bottom[threadIdx.y][col]
					HD = partial_bottom[seqId_block][(col_block % (threads_per_pair * 2))*8 + col_cell];
					h[0] = HD.x;
					e = HD.y;
//#pragma unroll 8
					for (l = 28 - 4*local_start, m = 1 + local_start; m < 1 + (local_end + 1); l -= 4, m++) {
						CORE_LOCAL_COMPUTE();

						row_max[seqId_block][tid * 8 + m - 1] = row_max[seqId_block][tid * 8 + m - 1] > h[m] ? row_max[seqId_block][tid * 8 + m - 1] : h[m];
						row_max_col[seqId_block][tid * 8 + m - 1] = row_max[seqId_block][tid * 8 + m - 1] > h[m] ? row_max_col[seqId_block][tid * 8 + m - 1] : global_col;

					}
					p[m] = h[m-1];
					//----------save intermediate values------------
					//save result on shared_memory[~][col]
					HD.x = h[m-1];
					HD.y = e;
					partial_bottom[seqId_block][(col_block % (threads_per_pair * 2))*8 + col_cell] = HD;
					if(global_col == read_len - 1){
						is_end_last = 1;
						break;
					}
					//---------------------------------------------------------
				}
				++col_block;
			}
			__syncwarp();
			++step;
			step_q = (step) / threads_per_pair;
			step_r = (step) % threads_per_pair;
			if(step_q >= 2 && step_r == 0){
				offset_partial_bottom = (step_q % 2) * threads_per_pair;
				offset_device_bottom = (step_q -2) * threads_per_pair;
				offset_device_bottom = (offset_device_bottom + tid) * 8;
				for(i = 0; i < 8; ++i){
					partial_bottom[seqId_block][(offset_partial_bottom + tid)*8 + i] = device_bottom[offset_device_bottom + threads_per_pair*2*8 + i];
				}
			}

			if(step % (threads_per_pair) == 0){
				bases[seqId_block][((step/threads_per_pair)%2)*threads_per_pair + tid] = packed_query_batch[packed_query_batch_idx + step + tid];
				//bases[seqId_block][threads_per_pair + tid] = packed_query_batch[packed_query_batch_idx + step + threads_per_pair + tid];
			}
			__syncwarp();
		}
		for(m = 1; m < 9; ++m){
			if(row_max[seqId_block][tid * 8 + m - 1] > thread_max[seqId_block][tid+1].x){
				thread_max[seqId_block][tid+1].x = row_max[seqId_block][tid * 8 + m - 1];
				thread_max[seqId_block][tid+1].y = global_row + m -1;
				thread_max[seqId_block][tid+1].z = row_max_col[seqId_block][tid * 8 + m - 1];
			}
		}
		//kogge-stone algorithm
		for(i = 1; i < threads_per_pair; i = i * 2){
			if(i <= tid && (thread_max[seqId_block][tid].x <= thread_max[seqId_block][tid-i].x)){
				SRCE.x = thread_max[seqId_block][tid-i].x;
				SRCE.y = thread_max[seqId_block][tid-i].y;
				SRCE.z = thread_max[seqId_block][tid-i].z;
				__syncwarp();
				thread_max[seqId_block][tid].x = SRCE.x;
				thread_max[seqId_block][tid].y = SRCE.y;
				thread_max[seqId_block][tid].z = SRCE.z;
			}
			else{
				__syncwarp();	
			}
			__syncwarp();
		}

		//check_early stop && z_drop && get accum max
		for(m = 1; m < 9; ++m){
			//early
			if(local_start <= (m-1) && (m-1) <= local_end && is_end_last){
				if(h[m] >= thread_g[seqId_block][tid+1].x && (global_row+m-1)<ref_len){
					thread_g[seqId_block][tid+1].x = h[m];
					thread_g[seqId_block][tid+1].y = global_row + m -1;
				} 
			}
			if(row_max[seqId_block][tid * 8 + m - 1] == 0){
				thread_max[seqId_block][tid].w = tid;
				break;
			}
			if(row_max[seqId_block][tid * 8 + m - 1] > thread_max[seqId_block][tid].x){
				thread_max[seqId_block][tid].x = row_max[seqId_block][tid * 8 + m - 1];
				thread_max[seqId_block][tid].y = global_row + m - 1;
				thread_max[seqId_block][tid].z = row_max_col[seqId_block][tid * 8 + m - 1];

				int max_off = (global_row + m -1) - row_max_col[seqId_block][tid * 8 + m - 1];
				if(max_off < 0) max_off = -max_off;
				thread_max_off[seqId_block][tid+1] = thread_max_off[seqId_block][tid+1] > max_off ? thread_max_off[seqId_block][tid+1] : max_off;
			}
			
			//z-drop
			//thread_max[seqId_block][tid].y,z ~ (global_row+m-1)~row_max_col[seqId_block][tid * 8 + m - 1];
			if((thread_max[seqId_block][tid].y - (global_row + m -1)) > (thread_max[seqId_block][tid].z - row_max_col[seqId_block][tid * 8 + m - 1])){
				if(thread_max[seqId_block][tid].x - row_max[seqId_block][tid * 8 + m - 1] - ((thread_max[seqId_block][tid].y - (global_row + m -1)) - (thread_max[seqId_block][tid].z - row_max_col[seqId_block][tid * 8 + m - 1]))*_cudaGapExtend > z_drop){
					thread_max[seqId_block][tid].w = tid;
					break;
				}
			}
			else{
				if(thread_max[seqId_block][tid].x - row_max[seqId_block][tid * 8 + m - 1] - ((thread_max[seqId_block][tid].z - row_max_col[seqId_block][tid * 8 + m - 1]) - (thread_max[seqId_block][tid].y - (global_row + m -1)))*_cudaGapExtend > z_drop){
					thread_max[seqId_block][tid].w = tid;
					break;
				}
			}
		}

		__syncwarp();
		//again kogge-stone algorithm for finding final maximum
		for(i = 1; i < threads_per_pair; i = i * 2){
			int temp = 0;
			if(i <= tid){
				temp = thread_max[seqId_block][tid-i].w < thread_max[seqId_block][tid].w ? thread_max[seqId_block][tid-i].w : thread_max[seqId_block][tid].w;
				__syncwarp();
				thread_max[seqId_block][tid].w = temp;
			}
			else{
				__syncwarp();	
			}
			__syncwarp();
		}


		//kogge-stone algorithm for max_off. target idx is 1<= < threads_per_pair+1
		for(i = 1; i < threads_per_pair + 1; i = i * 2){
			int bigger = 0;
			if(i-1 <= tid){
				bigger = thread_max_off[seqId_block][tid + 1-i] < thread_max_off[seqId_block][tid + 1] ? thread_max_off[seqId_block][tid + 1] : thread_max_off[seqId_block][tid + 1-i];
				__syncwarp();
				thread_max_off[seqId_block][tid + 1] = bigger;
			}
			else{
				__syncwarp();	
			}
			__syncwarp();
		}

		//kogge-stone for gscore
		for(i = 1; i < threads_per_pair + 1; i = i * 2){
			short2 big = initHD;
			if(i-1 <= tid){
				big.x = thread_g[seqId_block][tid + 1 - i].x;
				big.y = thread_g[seqId_block][tid + 1 - i].y;
				if(thread_g[seqId_block][tid + 1].x >= thread_g[seqId_block][tid + 1 - i].x){
					big.x = thread_g[seqId_block][tid + 1].x;
					big.y = thread_g[seqId_block][tid + 1].y;
				}
				__syncwarp();
				thread_g[seqId_block][tid+1].x = big.x;
				thread_g[seqId_block][tid+1].y = big.y;
			}
			else{
				__syncwarp();	
			}
			__syncwarp();
		}

		if(tid == 0){
			int idx = thread_max[seqId_block][threads_per_pair-1].w == threads_per_pair ? threads_per_pair -1 : thread_max[seqId_block][threads_per_pair-1].w;
			thread_max[seqId_block][0].x = thread_max[seqId_block][idx].x;
			thread_max[seqId_block][0].y = thread_max[seqId_block][idx].y;
			thread_max[seqId_block][0].z = thread_max[seqId_block][idx].z;
			thread_max[seqId_block][0].w = thread_max[seqId_block][idx].w;

			thread_max_off[seqId_block][0] = thread_max_off[seqId_block][idx + 1] > thread_max_off[seqId_block][0] ? thread_max_off[seqId_block][idx + 1] : thread_max_off[seqId_block][0];
			thread_g[seqId_block][0].y = thread_g[seqId_block][idx + 1].x >= thread_g[seqId_block][0].x ? thread_g[seqId_block][idx + 1].y : thread_g[seqId_block][0].y;
			thread_g[seqId_block][0].x = thread_g[seqId_block][idx + 1].x >= thread_g[seqId_block][0].x ? thread_g[seqId_block][idx + 1].x : thread_g[seqId_block][0].x; 
		}
		__syncwarp();

	//--------------------------------------------------------------------------------------------------------------------------------------
	if(tid == 0){
		device_res->aln_score[seqid] = thread_max[seqId_block][0].x;
		device_res->max_off[seqid] = thread_max_off[seqId_block][0];
		device_res->query_batch_end[seqid] = thread_max[seqId_block][0].z + 1;//copy the end position on query_batch sequence to the output array in the GPU mem
		device_res->target_batch_end[seqid] = thread_max[seqId_block][0].y + 1;//copy the end position on target_batch sequence to the output array in the GPU mem
		device_res->gscore[seqid] = thread_g[seqId_block][0].x; //gsocre
		device_res->max_ie[seqid] = thread_g[seqId_block][0].y + 1;
	}

	return; 
}
#endif
