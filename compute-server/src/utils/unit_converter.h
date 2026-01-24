/**
 * @file unit_converter.h
 * @brief Unit system conversion utilities
 */

#ifndef FEA_UNIT_CONVERTER_H
#define FEA_UNIT_CONVERTER_H

#include <string>
#include <map>

namespace FEA {

/**
 * @brief Supported unit systems
 */
enum class UnitSystem {
    SI,          // m, N, Pa, kg/m³
    SI_MM,       // mm, N, MPa, tonne/mm³
    US_CUSTOMARY // in, lbf, psi, slinch/in³
};

/**
 * @brief Unit converter for consistent units
 */
class UnitConverter {
public:
    explicit UnitConverter(UnitSystem system = UnitSystem::SI);
    
    /**
     * @brief Set unit system
     */
    void set_system(UnitSystem system);
    void set_system(const std::string &system_name);
    
    /**
     * @brief Get current unit system
     */
    UnitSystem get_system() const { return system_; }
    
    /**
     * @brief Convert length to SI (meters)
     */
    double length_to_si(double value) const;
    
    /**
     * @brief Convert length from SI
     */
    double length_from_si(double value) const;
    
    /**
     * @brief Convert force to SI (Newtons)
     */
    double force_to_si(double value) const;
    double force_from_si(double value) const;
    
    /**
     * @brief Convert stress/pressure to SI (Pascals)
     */
    double stress_to_si(double value) const;
    double stress_from_si(double value) const;
    
    /**
     * @brief Convert density to SI (kg/m³)
     */
    double density_to_si(double value) const;
    double density_from_si(double value) const;
    
    /**
     * @brief Get unit labels for current system
     */
    std::string length_unit() const;
    std::string force_unit() const;
    std::string stress_unit() const;
    std::string density_unit() const;
    
    /**
     * @brief Parse unit system from string
     */
    static UnitSystem parse_system(const std::string &name);

private:
    UnitSystem system_;
    
    // Conversion factors to SI
    double length_factor_;   // multiply to convert to meters
    double force_factor_;    // multiply to convert to Newtons
    double stress_factor_;   // multiply to convert to Pascals
    double density_factor_;  // multiply to convert to kg/m³
    
    void update_factors();
};

} // namespace FEA

#endif // FEA_UNIT_CONVERTER_H
