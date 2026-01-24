/**
 * @file inertial_loads.h
 * @brief Inertial load definitions (alias for body_loads.h)
 * 
 * This file provides backwards compatibility and convenience aliases
 * for inertial load types. The main implementations are in body_loads.h.
 */

#ifndef INERTIAL_LOADS_H
#define INERTIAL_LOADS_H

#include "body_loads.h"

namespace FEA {

// Re-export body load types under inertial load names for convenience
using InertialLoad = std::variant<GravityLoad, LinearAccelerationLoad, CentrifugalLoad>;

// Type aliases for consistency
using RotationalLoad = CentrifugalLoad;
using AccelerationLoad = LinearAccelerationLoad;

/**
 * @brief Create a centrifugal load from angular velocity in rad/s
 */
inline CentrifugalLoad create_centrifugal_load(
    const Point<3>& axis_point,
    const Tensor<1, 3>& axis_direction,
    double omega_rad_s) {
    return CentrifugalLoad(axis_point, axis_direction, omega_rad_s);
}

/**
 * @brief Create a centrifugal load from angular velocity in RPM
 */
inline CentrifugalLoad create_centrifugal_load_rpm(
    const Point<3>& axis_point,
    const Tensor<1, 3>& axis_direction,
    double rpm) {
    return CentrifugalLoad::from_rpm(axis_point, axis_direction, rpm);
}

/**
 * @brief Create a gravity load with custom acceleration
 */
inline GravityLoad create_gravity_load(double gx, double gy, double gz) {
    return GravityLoad::custom(gx, gy, gz);
}

/**
 * @brief Create standard Earth gravity load (-9.81 m/s² in Z)
 */
inline GravityLoad create_standard_gravity() {
    return GravityLoad::standard();
}

/**
 * @brief Create a linear acceleration load
 */
inline LinearAccelerationLoad create_acceleration_load(double ax, double ay, double az) {
    return LinearAccelerationLoad::create(ax, ay, az);
}

} // namespace FEA

#endif // INERTIAL_LOADS_H
