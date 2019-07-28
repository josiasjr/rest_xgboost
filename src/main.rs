//LIBRARY_PATH=/usr/lib;LD_LIBRARY_PATH=/usr/lib
use actix_web::{web, App, HttpRequest, HttpResponse, HttpServer, Responder, Result};
use serde_json::{Value, json};
use std::ffi::c_void;
mod xgboost;
use xgboost::XGBoost;
use std::ptr;
use std::time::{SystemTime, UNIX_EPOCH};

const COLUMNS:[&str; 30] = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"];
static mut BST:*mut c_void = ptr::null_mut();

fn greet(_: HttpRequest) -> impl Responder {
	let js = json!({"hello":"world"});
	js.to_string()
}

fn get_system_time() -> f64 {SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as f64 / 1e6}

fn calculator(json: web::Json<Value>) -> Result<HttpResponse> {
	let start = get_system_time();
	let js = COLUMNS.iter().map(|x| json[x].as_f64().unwrap() as f32).collect::<Vec<f32>>();
	let x_test = XGBoost::create_dmatrix(&js, 1);
	let res = unsafe{XGBoost::predict(&BST, &x_test)[0]};
	let js = json!({"score":res, "elapsed_time": get_system_time() - start});
	Ok(HttpResponse::Ok()
		.content_type("application/json")
//		.header("X-Hdr", "sample")
		.body(js.to_string()))
}

fn main() {
	unsafe {BST = XGBoost::load_model("/home/josias/workspace/IdeaProjects/bi/analytics/kaggle_crd_fraud/model.xgb")};
	HttpServer::new(|| {
		App::new()
			.route("/", web::get().to(greet))
			.route("/calculator/predict", web::post().to(calculator))
	})
		.workers(num_cpus::get())
		.bind("0.0.0.0:8080")
		.expect("Can not bind to port 8000")
		.run()
		.unwrap();
}
