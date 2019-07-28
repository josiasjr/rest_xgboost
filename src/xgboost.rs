//export LD_LIBRARY_PATH=/usr/lib/
//export LIBRARY_PATH=/usr/lib/
use std::{fs, ptr, slice};
use std::ffi::c_void;

type BoosterHandle = *mut c_void;
type BstUlong = u64;
type DMatrixHandle = *mut c_void;

#[link(name = "xgboost")]
extern {
	fn XGDMatrixCreateFromMat(data:*const f32, nrow:BstUlong, ncol:BstUlong, missing:f32, out:*mut DMatrixHandle);//int XGDMatrixCreateFromMat(const float *data,bst_ulong nrow,bst_ulong ncol,float missing,DMatrixHandle *out);
	fn XGBoosterCreate(dmats:*const DMatrixHandle, len:BstUlong, out:*mut BoosterHandle) -> i32;//int XGBoosterCreate(const DMatrixHandle dmats[],bst_ulong len,BoosterHandle *out);
	fn XGBoosterLoadModelFromBuffer(handle:BoosterHandle, buf:*const c_void, len:BstUlong ) -> i32;//int XGBoosterLoadModelFromBuffer(BoosterHandle handle,const void *buf,bst_ulong len);
	fn XGBoosterPredict(handle:BoosterHandle, dmat:DMatrixHandle, option_mask:i32, ntree_limit:u32, out_len:*mut BstUlong, out_result:*mut*const f32) -> i32;//XGBoosterPredict(BoosterHandle handle,DMatrixHandle dmat,int option_mask,unsigned ntree_limit,bst_ulong *out_len,const float **out_result);
}

pub struct XGBoost{}
impl XGBoost{
	pub fn create_dmatrix(data: &[f32], num_rows: usize) -> DMatrixHandle{
		let mut handle_dmat = ptr::null_mut();
		unsafe{XGDMatrixCreateFromMat(data.as_ptr(), num_rows as BstUlong, (data.len()/num_rows) as BstUlong, 0.0, &mut handle_dmat);}
		handle_dmat
	}

	pub fn load_model(path: &str) -> BoosterHandle{
		let file = fs::read(path).expect("Error in loading file model");
		let mut handle_booster = ptr::null_mut();
		unsafe{
			XGBoosterCreate(ptr::null(), 0, &mut handle_booster);
			XGBoosterLoadModelFromBuffer(handle_booster, file.as_ptr() as *const _, file.len() as BstUlong);
		}
		handle_booster
	}

	pub fn predict(booster:&BoosterHandle, dmat:&DMatrixHandle) -> Vec<f32>{
		//option_mask	bit-mask of options taken in prediction, possible values 0:normal prediction 1:output margin instead of transformed value 2:output leaf index of trees instead of leaf value, note leaf index is unique per tree 4:output feature contributions to individual predictions
		//ntree_limit	limit number of trees used for prediction, this is only valid for boosted trees when the parameter is set to 0, we will use all the trees
		let mut out_len = 0;
		let mut out_result = ptr::null();
		unsafe {XGBoosterPredict(*booster, *dmat, 0, 0, &mut out_len, &mut out_result)};
		let data = unsafe {slice::from_raw_parts(out_result, out_len as usize).to_vec()};
		data
	}
}
