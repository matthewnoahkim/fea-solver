/**
 * @file unit_converter.cc
 * @brief Implementation of unit converter
 */

#include "unit_converter.h"
#include <stdexcept>
#include <algorithm>

namespace FEA {

UnitConverter::UnitConverter(UnitSystem system)
    : system_(system)
{
    update_factors();
}

void UnitConverter::set_system(UnitSystem system) {
    system_ = system;
    update_factors();
}

void UnitConverter::set_system(const std::string &system_name) {
    system_ = parse_system(system_name);
    update_factors();
}

UnitSystem UnitConverter::parse_system(const std::string &name) {
    std::string upper = name;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    
    if (upper == "SI" || upper == "METRIC") {
        return UnitSystem::SI;
    } else if (upper == "SI_MM" || upper == "MM" || upper == "METRIC_MM") {
        return UnitSystem::SI_MM;
    } else if (upper == "US_CUSTOMARY" || upper == "IMPERIAL" || upper == "US" || upper == "INCH") {
        return UnitSystem::US_CUSTOMARY;
    }
    
    throw std::invalid_argument("Unknown unit system: " + name);
}

void UnitConverter::update_factors() {
    switch (system_) {
        case UnitSystem::SI:
            length_factor_ = 1.0;           // m
            force_factor_ = 1.0;            // N
            stress_factor_ = 1.0;           // Pa
            density_factor_ = 1.0;          // kg/m³
            break;
            
        case UnitSystem::SI_MM:
            length_factor_ = 1e-3;          // mm -> m
            force_factor_ = 1.0;            // N
            stress_factor_ = 1e6;           // MPa -> Pa
            density_factor_ = 1e12;         // tonne/mm³ -> kg/m³
            break;
            
        case UnitSystem::US_CUSTOMARY:
            length_factor_ = 0.0254;        // in -> m
            force_factor_ = 4.44822;        // lbf -> N
            stress_factor_ = 6894.76;       // psi -> Pa
            density_factor_ = 175126.8;     // slinch/in³ -> kg/m³
            break;
    }
}

double UnitConverter::length_to_si(double value) const {
    return value * length_factor_;
}

double UnitConverter::length_from_si(double value) const {
    return value / length_factor_;
}

double UnitConverter::force_to_si(double value) const {
    return value * force_factor_;
}

double UnitConverter::force_from_si(double value) const {
    return value / force_factor_;
}

double UnitConverter::stress_to_si(double value) const {
    return value * stress_factor_;
}

double UnitConverter::stress_from_si(double value) const {
    return value / stress_factor_;
}

double UnitConverter::density_to_si(double value) const {
    return value * density_factor_;
}

double UnitConverter::density_from_si(double value) const {
    return value / density_factor_;
}

std::string UnitConverter::length_unit() const {
    switch (system_) {
        case UnitSystem::SI: return "m";
        case UnitSystem::SI_MM: return "mm";
        case UnitSystem::US_CUSTOMARY: return "in";
    }
    return "m";
}

std::string UnitConverter::force_unit() const {
    switch (system_) {
        case UnitSystem::SI:
        case UnitSystem::SI_MM: return "N";
        case UnitSystem::US_CUSTOMARY: return "lbf";
    }
    return "N";
}

std::string UnitConverter::stress_unit() const {
    switch (system_) {
        case UnitSystem::SI: return "Pa";
        case UnitSystem::SI_MM: return "MPa";
        case UnitSystem::US_CUSTOMARY: return "psi";
    }
    return "Pa";
}

std::string UnitConverter::density_unit() const {
    switch (system_) {
        case UnitSystem::SI: return "kg/m³";
        case UnitSystem::SI_MM: return "tonne/mm³";
        case UnitSystem::US_CUSTOMARY: return "slinch/in³";
    }
    return "kg/m³";
}

} // namespace FEA
