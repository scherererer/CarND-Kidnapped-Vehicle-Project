/*
 * particle_filter.h
 *
 * 2D particle filter class.
 */

#pragma once

#include "helper_functions.h"

#include <random>
#include <string>
#include <vector>


struct Particle {

	int id;
	double x;
	double y;
	double theta;
	double weight;
	std::vector<int> associations;
	std::vector<double> sense_x;
	std::vector<double> sense_y;
};



class ParticleFilter {

public:
	~ParticleFilter();
	// @param M Number of particles
	ParticleFilter();

	// Set of current particles
	std::vector<Particle> const &particles () const
		{ return particles_; }

	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std_dev[] Array of dimension 3 [standard deviation of x [m], standard deviation
	 *   of y [m], standard deviation of yaw [rad]]
	 */
	void init(double x, double y, double theta, double std_dev[]);

	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_dev[] Array of dimension 3 [standard deviation of x [m], standard
	 *   deviation of y [m], standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
	void prediction(double delta_t, double std_dev[], double velocity, double yaw_rate);

	/**
	 * dataAssociation Finds which observations correspond to which landmarks (likely by using
	 *   a nearest-neighbors data association).
	 * @param predicted Vector of predicted landmark observations
	 * @param observations Vector of landmark observations
	 */
	/// \todo Why is observations passed by reference like this? Not clear why it is supposed
	/// to be used as return value.
	std::vector<LandmarkObs> dataAssociation(std::vector<LandmarkObs> const &predicted,
	                                         std::vector<LandmarkObs> const &observations);

	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the
	 *   observed measurements.
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [standard deviation of x [m],
	 *   standard deviation of y [m]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */
	void updateWeights(double sensor_range, double std_landmark[],
	                   std::vector<LandmarkObs> observations, Map map_landmarks);

	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
	void resample();

	/*
	 * Set a particles list of associations, along with the associations calculated world x,y
	 * coordinates
	 * This can be a very useful debugging tool to make sure transformations are correct and
	 * assocations correctly connected
	 */
	Particle SetAssociations(Particle particle, std::vector<int> associations,
	                         std::vector<double> sense_x, std::vector<double> sense_y);

	std::string getAssociations(Particle best);
	std::string getSenseX(Particle best);
	std::string getSenseY(Particle best);

	/**
	 * initialized Returns whether particle filter is initialized yet or not.
	 */
	const bool initialized() const {
		return is_initialized_;
	}

private:
	// Number of particles to draw
	int const num_particles_;

	// Flag, if filter is initialized
	bool is_initialized_;

	// Set of current particles
	std::vector<Particle> particles_;

	// Vector of weights of all particles
	std::vector<double> weights_;

	std::random_device rand_device_;
    std::default_random_engine rand_engine_;
};

