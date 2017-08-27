#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ << 0.0043, -0.0013 ,0.0030 , -0.0022, -0.0020,
        -0.0013, 0.0077 ,0.0011 , 0.0071, 0.0060,
        0.0030,  0.0011 ,0.0054 , 0.0007, 0.0008,
        -0.0022, 0.0071 ,0.0007 , 0.0098, 0.0100,
        -0.0020, 0.0060 ,0.0008 , 0.0100, 0.0123;

  n_x_ = 5;
  n_aug_ = 7;
  n_sig_ = 2*n_aug_ + 1;
  lambda_ = 3 - n_aug_;

  Xsig_pred_ = MatrixXd(n_x_,n_sig_);  

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  //create vector for weights
  weights_ =VectorXd(n_sig_);
  weights_(0) = lambda_/(lambda_ + n_aug_);
  for (int i; i < n_sig_; i++){
  	weights_(i) = 0.5/(n_aug_ + lambda_);
  }

  is_initialized_ = false;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (! is_initialized_) {
  	time_us_ = meas_package.timestamp_;
  	if (meas_package.sensor_type_ == MeasurementPackage::LASER){
  		float px = meas_package.raw_measurements_[0];
  		float py = meas_package.raw_measurements_[1];

  		//CTRV Model State Vector

  		x_ << px, py, 0, 0, 0;
  	}
  	else if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
  		float rho = meas_package.raw_measurements_[0];
  		float phi = meas_package.raw_measurements_[1];
  		float rho_dot = meas_package.raw_measurements_[2];

  		float px = rho * cos(phi);
  		float py = rho * sin(phi);

  		//CTRV Model State Vector

  		x_ << px, py, 0, 0, 0;

  	}
  	is_initialized_ = true;

}
  else {

  	//prediction
  	double delta_t = (meas_package.timestamp_ - time_us_) /1000000.0;
  	time_us_ = meas_package.timestamp_;

  	Prediction(delta_t);

  	if(meas_package.sensor_type_ == MeasurementPackage::LASER){

  		UpdateLidar(meas_package);
  	}
  	else if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
  		UpdateRadar(meas_package);
  	} 
  }


}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  VectorXd x_aug_(7);
  x_aug_.fill(0);
  x_aug_.head(n_x_) = x_;

  // create 7 x 7 augmented covariance matrix

  MatrixXd P_aug_(n_aug_, n_aug_);
  P_aug_.fill(0);
  P_aug_.topLeftCorner(n_x_,n_x_) = P_;
  P_aug_(5,5) = pow(std_a_, 2);
  P_aug_(6,6) = pow(std_yawdd_, 2);

  // create square root matrix
  MatrixXd L_ = P_aug_.llt().matrixL();

  // create augmented sigma points

  MatrixXd Xsig_aug_(n_aug_,n_sig_);

  // first column
  Xsig_aug_.col(0) = x_aug_;
  for (int i = 0; i < n_aug_; i ++){
  	Xsig_aug_.col(i+1)         = x_aug_    +  sqrt(lambda_ + n_aug_) * L_.col(i);
  	Xsig_aug_.col(i+1+n_aug_)  = x_aug_    -  sqrt(lambda_ + n_aug_) * L_.col(i);

  }

  // Sigma point prediction

  for(int i = 0; i < n_sig_; i ++) {

  	// Sigma point prediction, from 7 x 15 to 5 x 15 matrix

  	double p_x    = Xsig_aug_.col(i)(0);
  	double p_y    = Xsig_aug_.col(i)(1);
  	double v      = Xsig_aug_.col(i)(2);
  	double yaw    = Xsig_aug_.col(i)(3);
  	double yawd   = Xsig_aug_.col(i)(4);
  	double nu_a   = Xsig_aug_.col(i)(5);
  	double nu_yawdd = Xsig_aug_.col(i)(6);

  	double px_p, py_p;
  	if(fabs(yawd) > 0.001){
  		px_p  =  p_x  +  v/yawd * (sin(yaw + yawd*delta_t)  - sin(yaw));
  		py_p  =  p_y  +  v/yawd * (cos(yaw)    - cos(yaw+yawd*delta_t));
  	}else{
  		px_p  =  p_x  +  v * delta_t * cos(yaw);
  		py_p  =  p_y  +  v * delta_t * sin(yaw);

  	}

  	double v_p = v;
  	double yaw_p = yaw + yawd * delta_t;
  	double yawd_p = yawd;

  	// add noise for acceleration and yaw rate(nu_a, nu_yawdd)

  	px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
  	py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
  	v_p  = v_p  + nu_a * delta_t;

  	yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
  	yaw_p = yawd_p + nu_yawdd * delta_t;

  	Xsig_pred_(0,i) = px_p;
  	Xsig_pred_(1,i) = py_p;
  	Xsig_pred_(2,i) = v_p;
  	Xsig_pred_(3,i) = yaw_p;
  	Xsig_pred_(4,i) = yawd_p;


  }

  //predicted state mean

  x_ = Xsig_pred_ * weights_;

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sig_; i++){
  	VectorXd x_diff_ = Xsig_pred_.col(i) - x_;

  	//Normalizing angle
  	while (x_diff_(3) > M_PI)  x_diff_(3) -= 2.*M_PI;
  	while (x_diff_(3) > -M_PI) x_diff_(3) += 2.*M_PI;

  	P_  = P_ + weights_(i) * x_diff_.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  // transform sigma points into 2 x 15 measurement space

  MatrixXd Zsig_  = MatrixXd(2,15);
  VectorXd z_pred_  = VectorXd(2);

  // measurement model

  Zsig_.row(0) = Xsig_pred_.row(0);           //px
  Zsig_.row(1) = Xsig_pred_.row(1);           //py

  z_pred_      = Zsig_ * weights_;

  // measurement covariance matrix s

  MatrixXd S_ = MatrixXd(2,2);
  S_.fill(0);

  // create matrix for cross correlation Tc
  MatrixXd Tc_ = MatrixXd(5,2);
  Tc_.fill(0);

  for (int i = 0; i < n_sig_; i++) {

  	//measurement residual
  	VectorXd tz_diff_  = Zsig_.col(i) - z_pred_;
  	S_ = S_ + weights_(i) * tz_diff_ * tz_diff_.transpose();

  	//state residual
  	VectorXd tx_diff_  = Xsig_pred_.col(i) - x_;
  	Tc_ = Tc_ + weights_(i) * tx_diff_ * tz_diff_.transpose();
  }

  // Laser measurement noise covariance matrix
  MatrixXd R_laser_(2,2);
  R_laser_  <<  std_laspx_ * std_laspx_ , 0,
                0, std_laspy_ *std_laspy_;

  // add measurement noise covariance matrix

  S_ = S_ + R_laser_;

  // calculate kalman gain K

  MatrixXd K_ = Tc_ * (S_.inverse());

  // Actual measurement against predicted measurement

  VectorXd A_ = meas_package.raw_measurements_ - z_pred_;

  // update state

  x_ = x_ + K_ * A_;

  // update NIS

  NIS_laser_  = (A_.transpose())*S_.inverse()*A_;

  // Update stat covariance

  P_ = P_ - K_* S_* K_.transpose();

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  // Transfrom sigma points into 3 x 15 measurement space
  MatrixXd Zsig_ = MatrixXd(3, n_sig_);

  // predicted measurement mean

  VectorXd z_pred_ = VectorXd(3);

  // copy the sigma point predictions

  for (int i = 0; i < n_sig_; i++){
  	double px =  Xsig_pred_(0,i);
  	double py =  Xsig_pred_(1,i);
  	double v  =  Xsig_pred_(2,i);
  	double yaw = Xsig_pred_(3,i);
  	double yawd = Xsig_pred_(4,i);

  	double vx = cos(yaw) * v;
  	double vy = sin(yaw) * v;

  	// predicted measurement rho

  	Zsig_(0,i)   =  sqrt(px*px+py*py);   //rho

  	// predicted measurement phi
  	Zsig_(1,i)   =  atan2(py,px);

  	// predicted measurement rho_dot

  	if(Zsig_(0,i) > 0.001){
  		Zsig_(2,i) = (px*vx + py*vy)/sqrt(px*px+py*py);  //rho_dot
  	} else {

  		Zsig_(2,i) = 0.0;         // rho_dit
  	}
  }

  // calculate predicted measurement mean

  z_pred_ = Zsig_ * weights_;

  // Create measurement covariance matrix S
  MatrixXd S_ = MatrixXd(3,3);
  S_.fill(0);

  // Create matrix for cross correlation Tc
  MatrixXd Tc_ = MatrixXd(5,3);
  Tc_.fill(0);

  for (int i=0; i < n_sig_; i++) {
  	// Calculate the difference
  	VectorXd z_diff_ = Zsig_.col(i) - z_pred_;
  	// Calculate 3 x 3 predicted measurement covariance S_
  	S_ = S_ + weights_(i) * z_diff_ * z_diff_.transpose();

  	// Calculate state residual
  	VectorXd tx_diff_ = Xsig_pred_.col(i) - x_;

  	// Calcualte measurement residual
  	VectorXd tz_diff_ = Zsig_.col(i) - z_pred_;
  	Tc_ = Tc_ + weights_(i) * tx_diff_*tz_diff_.transpose();
  }

  // Create Radar Measurement Noise Covariance Matrix

  MatrixXd R_radar_(3,3);
  R_radar_ << std_radr_ * std_radr_ , 0, 0,
                0, std_radphi_ * std_radphi_,0,
                0,0,std_radrd_ * std_radrd_;

  // add measurment noise covariance matrix

  S_ = S_ + R_radar_;

  // Calculate Kalman gain K

  MatrixXd K_ = Tc_ * (S_.inverse());

  // Actual measurement against predicted measurement

  VectorXd A_ = meas_package.raw_measurements_ - z_pred_;

  //update state

  x_ = x_ + K_ * A_;

  // update NIS

  NIS_radar_ = (A_.transpose())*S_.inverse()*A_;

  //update state covariance

  P_ = P_ - K_ * S_ * K_.transpose();
}
