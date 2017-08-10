/*
 * particle_filter.cpp
 */

#include "particle_filter.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
using namespace std;


namespace
{

inline double angleWrap (double a_)
{
	a_ = fmod (a_ + M_PI, 2.0 * M_PI);
	if (a_ < 0)
		a_ += M_PI * 2.0;
	return a_ - M_PI;
}

template<typename T>
inline T clip (T value, T min, T max)
{
	return std::min(max, std::max(min, value));
}
}


ParticleFilter::~ParticleFilter()
{
}

ParticleFilter::ParticleFilter()
	: num_particles_(100)
	, is_initialized_(false)
	, particles_ ()
	, weights_ (num_particles_, 0.0)
	, rand_device_ ()
	, rand_engine_ (rand_device_ ())
{
}

void ParticleFilter::init(double x, double y, double theta, double std_dev[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on
	// estimates of x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in
	// this file).

	std::normal_distribution<double> randX(x, std_dev[0]);
	std::normal_distribution<double> randY(y, std_dev[1]);
	std::normal_distribution<double> randTheta(theta, std_dev[2]);

	double const initialWeight = 1.0;

	for (unsigned i = 0; i < num_particles_; ++i)
	{
		Particle p;

		p.id = i;
		p.x = randX(rand_engine_);
		p.y = randY(rand_engine_);
		p.theta = angleWrap(randTheta(rand_engine_));

		p.weight = initialWeight;

		particles_.push_back(p);
	}

	is_initialized_ = true;
}

void ParticleFilter::prediction(double delta_t, double std_dev[], double velocity,
                                double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and
	// std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::normal_distribution<double> randX(0.0, std_dev[0]);
	std::normal_distribution<double> randY(0.0, std_dev[1]);
	std::normal_distribution<double> randTheta(0.0, std_dev[2]);

	for (Particle &p : particles_)
	{
		double const thetaf = p.theta + yaw_rate * delta_t;
		double const xf =
			p.x + velocity / yaw_rate * (sin(thetaf) - sin(p.theta));
		double const yf =
			p.y + velocity / yaw_rate * (cos(p.theta) - cos(thetaf));

		p.x = xf + randX(rand_engine_);
		p.y = yf + randY(rand_engine_);
		p.theta = angleWrap(thetaf + randTheta(rand_engine_));
	}
}

vector<LandmarkObs> ParticleFilter::dataAssociation(vector<LandmarkObs> const &predicted,
													vector<LandmarkObs> const &observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and
	// assign the observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it
	// useful to implement this method and use it as a helper during the updateWeights phase.

	assert(! predicted.empty());

	std::vector<LandmarkObs> associations;

	associations.reserve (observations.size ());

	for (LandmarkObs const &obs : observations)
	{
		LandmarkObs nearest;
		double nearestSquareDist = numeric_limits<double>::max ();

		for (LandmarkObs const &pred : predicted)
		{
			double const dx = pred.x - obs.x;
			double const dy = pred.y - obs.y;
			double const squareDist = dx * dx + dy * dy;

			if (squareDist < nearestSquareDist)
			{
				nearest = pred;
				nearestSquareDist = squareDist;
			}
		}

		assert(nearestSquareDist != numeric_limits<double>::max ());

		associations.push_back(nearest);
	}

	return associations;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution.
	// You can read more about this distribution here:
	//   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are
	//   located according to the MAP'S coordinate system. You will need to transform between
	//   the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no
	//   scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at
	//   equation 3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// Optimization so we don't have to take the square root
	double const sensorRangeSquared = sensor_range * sensor_range;

	for (unsigned p = 0; p < num_particles_; ++p)
	{
		Particle const &particle = particles_[p];
		std::vector<LandmarkObs> predictedLandmarks; // Filtered landmarks
		std::vector<LandmarkObs> transformedObservations; // Transformed observations

		// Add and transform all observations into map space
		for (LandmarkObs const &obs : observations)
		{
			LandmarkObs to;

			/// \todo I may need to reverse this transform
			double const ct = cos(particle.theta);
			double const st = sin(particle.theta);

			to.id = obs.id;
			to.x = obs.x * ct - obs.y * st + particle.x;
			to.y = obs.x * st + obs.y * ct + particle.y;

			transformedObservations.push_back (to);
		}

		// Collect all landmarks that are in range
		for (Map::single_landmark_s const &l : map_landmarks.landmark_list)
		{
			double const dx = l.x_f - particle.x;
			double const dy = l.y_f - particle.y;

			if (dx * dx + dy * dy > sensorRangeSquared)
				continue;

			LandmarkObs fl;

			fl.id = l.id_i;
			fl.x = l.x_f;
			fl.y = l.y_f;

			predictedLandmarks.push_back (fl);
		}

		/// \todo I'm not sure if this is the best thing to do -- but if it can't make any
		/// associations then it certainly can't make any correct associations, so give it no
		/// weight
		if (predictedLandmarks.empty () || transformedObservations.empty ())
		{
			cerr << "No predictions in range\n";
			weights_[p] = 0.0;
			continue;
		}

		/// \todo Find associations
		vector<LandmarkObs> const associations =
			dataAssociation (predictedLandmarks, transformedObservations);

		/// \todo Calculate probability for the measurement
		double weight = 1.0;

		for (unsigned i = 0; i < transformedObservations.size (); ++i)
		{
			double const sigma_x = std_landmark[0];
			double const sigma_y = std_landmark[1];

			double const x = transformedObservations[i].x;
			double const y = transformedObservations[i].y;
			double const u_x = associations[i].x;
			double const u_y = associations[i].y;

			double const dx = x - u_x;
			double const dy = y - u_y;

			double const xterm = (dx * dx) / (2 * sigma_x * sigma_x);
			double const yterm = (dy * dy) / (2 * sigma_y * sigma_y);

			weight *= exp(-1.0 * (xterm + yterm)) / (2 * M_PI * sigma_x * sigma_y);
		}

		weights_[p] = weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	assert (weights_.size() == num_particles_);
	assert (particles_.size() == num_particles_);

	std::discrete_distribution<int> dd(weights_.begin (), weights_.end ());
	std::vector<Particle> newParticles;

	newParticles.reserve (num_particles_);

	for (unsigned i = 0; i < num_particles_; ++i)
		newParticles.push_back(particles_[dd(rand_engine_)]);

	assert (particles_.size() == newParticles.size());
	particles_.swap(newParticles);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations,
                                         std::vector<double> sense_x,
                                         std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world
	//coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
