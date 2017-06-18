/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	default_random_engine generator;
	normal_distribution<double> dist_x(0, std[0]);
	normal_distribution<double> dist_y(0, std[1]);
	normal_distribution<double> dist_theta(0, std[2]);

	num_particles = 200;

	for (int i = 0; i < num_particles; i += 1) {
		Particle p;
		p.id = i;
		p.x = x;
		p.y = y;
		p.theta = theta;

		p.x += dist_x(generator);
		p.y += dist_y(generator);
		p.theta += dist_theta(generator);

		p.weight = 1.0f;
		particles.push_back(p);
		weights.push_back(1.0f);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine generator;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; i += 1) {
		// Handle division by zero error
		if (fabs(yaw_rate) < 0.01) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
		} else {
			particles[i].x +=
				velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y +=
				velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
		}
		particles[i].theta += yaw_rate * delta_t;
		particles[i].x += dist_x(generator);
		particles[i].y += dist_y(generator);
		particles[i].theta += dist_theta(generator);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i = 0; i < observations.size(); i += 1) {
		LandmarkObs& observation = observations[i];
		double min_distance = INFINITY;
		int landmark_id = -1;
		for (unsigned int j = 0; j < predicted.size(); j += 1) {
			LandmarkObs prediction = predicted[j];
			double distance = dist(observation.x, observation.y, prediction.x, prediction.y);
			if (distance < min_distance) {
				min_distance = distance;
				landmark_id = prediction.id;
			}
		}
		observation.id = landmark_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	//
	for (int i = 0; i < num_particles; i += 1) {
		Particle& particle = particles[i];
		// Vector with landmarks within range
		vector<LandmarkObs> predictions;

		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j += 1) {
			float landmark_x = map_landmarks.landmark_list[j].x_f;
			float landmark_y = map_landmarks.landmark_list[j].y_f;
			int landmark_id = map_landmarks.landmark_list[j].id_i;
			// Consider all landmarks within a square sensor range (for fast computing)
			if (fabs(landmark_x - particle.x) <= sensor_range && fabs(landmark_y - particle.y) <= sensor_range) {
				predictions.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
			}
		}

		// Vector with landmarks transformed to map coordinates
		vector<LandmarkObs> transformed_to_map;
		for (int j = 0; j < observations.size(); j += 1) {
			double trans_x =
				cos(particle.theta) * observations[j].x - sin(particle.theta) * observations[j].y + particle.x;
			double trans_y =
				sin(particle.theta) * observations[j].x + cos(particle.theta) * observations[j].y + particle.y;
			transformed_to_map.push_back(LandmarkObs{ j, trans_x, trans_y });
		}

		// Get associated data
		dataAssociation(predictions, transformed_to_map);

		// reinitiate weight
		double total_prob = 1.0;

		for (unsigned int j = 0; j < transformed_to_map.size(); j += 1) {
			double observation_x, observation_y, prediction_x, prediction_y;
			observation_x = transformed_to_map[j].x;
			observation_y = transformed_to_map[j].y;
			int prediction_associated = transformed_to_map[j].id;

			// Find the associated prediction
			for (unsigned int k = 0; k < predictions.size(); k += 1) {
				if (predictions[k].id == prediction_associated) {
					prediction_x = predictions[k].x;
					prediction_y = predictions[k].y;
					break;
				}
			}

			double s_x = std_landmark[0];
			double s_y = std_landmark[1];
			double prob =
				(1 / (2 * M_PI * s_x * s_y)) * exp( -( pow(prediction_x - observation_x, 2) / (2 * pow(s_x, 2)) + (pow(prediction_y - observation_y, 2) / (2 * pow(s_y, 2))) ) );
			// cout << "predictions: " << predictions.size() << endl;
			// cout << "Prediction_x: " << prediction_x << "; prediction_y: " << prediction_y << "; observation_x: " << observation_x << "observation_y: " << observation_y << endl;
			total_prob *= prob;
		}
		particle.weight = total_prob;
		weights[i] = total_prob;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine generator;
	vector<Particle> resampled;
	vector<double> weights;

	for (int i = 0; i < num_particles; i += 1) {
		weights.push_back(particles[i].weight);
	}

	// Random start point for wheel
	uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(generator);

	// Get highest weight
	double max_weight = *max_element(weights.begin(), weights.end());
	uniform_real_distribution<double> unirealdist(0.0, max_weight);
	double beta = 0.0;

	for (int i = 0; i < num_particles; i += 1) {
		beta += unirealdist(generator) * 2.0;
		while (beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		resampled.push_back(particles[index]);
	}
	particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
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
