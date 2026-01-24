/**
 * @file material_library.h
 * @brief Complete material model system for FEA
 * 
 * Supports:
 * - Isotropic linear elastic (steel, aluminum, etc.)
 * - Orthotropic linear elastic (composites, wood)
 * - Elastoplastic with von Mises yield (ductile metals beyond yield)
 * - Hyperelastic Neo-Hookean and Mooney-Rivlin (rubber, elastomers)
 */

#ifndef MATERIAL_LIBRARY_H
#define MATERIAL_LIBRARY_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/full_matrix.h>
#include <string>
#include <variant>
#include <optional>
#include <map>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace FEA {

using namespace dealii;

// ============================================================================
// Material Model Enumeration
// ============================================================================

enum class MaterialModel {
    LINEAR_ELASTIC_ISOTROPIC,
    LINEAR_ELASTIC_ORTHOTROPIC,
    ELASTOPLASTIC_VONMISES,
    ELASTOPLASTIC_DRUCKER_PRAGER,
    HYPERELASTIC_NEOHOOKEAN,
    HYPERELASTIC_MOONEY_RIVLIN
};

std::string material_model_to_string(MaterialModel model);
MaterialModel string_to_material_model(const std::string& str);

// ============================================================================
// Isotropic Linear Elastic Properties
// ============================================================================

struct IsotropicElasticProperties {
    double youngs_modulus;          // E [Pa]
    double poissons_ratio;          // nu [-], must be in (-1, 0.5)
    double density;                 // rho [kg/m³]
    double thermal_expansion_coeff; // alpha [1/K]
    
    // Default constructor
    IsotropicElasticProperties()
        : youngs_modulus(0), poissons_ratio(0), density(0), 
          thermal_expansion_coeff(0) {}
    
    // Parameterized constructor
    IsotropicElasticProperties(double E, double nu, double rho, double alpha = 0)
        : youngs_modulus(E), poissons_ratio(nu), density(rho),
          thermal_expansion_coeff(alpha) {}
    
    // Lamé's first parameter: λ = E*ν / ((1+ν)(1-2ν))
    double lambda() const {
        return (youngs_modulus * poissons_ratio) / 
               ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio));
    }
    
    // Lamé's second parameter (shear modulus): μ = G = E / (2(1+ν))
    double mu() const {
        return youngs_modulus / (2.0 * (1.0 + poissons_ratio));
    }
    
    // Bulk modulus: K = E / (3(1-2ν))
    double bulk_modulus() const {
        return youngs_modulus / (3.0 * (1.0 - 2.0 * poissons_ratio));
    }
    
    // Build the 4th order elasticity tensor C_ijkl
    // σ_ij = C_ijkl * ε_kl
    // For isotropic: C_ijkl = λ*δ_ij*δ_kl + μ*(δ_ik*δ_jl + δ_il*δ_jk)
    SymmetricTensor<4, 3> get_elasticity_tensor() const;
    
    // Get thermal expansion tensor (isotropic)
    SymmetricTensor<2, 3> get_thermal_expansion_tensor() const;
    
    // Validate material properties
    bool is_valid() const {
        return youngs_modulus > 0 &&
               poissons_ratio > -1.0 &&
               poissons_ratio < 0.5 &&
               density >= 0;
    }
};

// ============================================================================
// Orthotropic Linear Elastic Properties
// ============================================================================

// For materials with three mutually perpendicular planes of symmetry
// Examples: unidirectional composites, wood, rolled sheet metal
struct OrthotropicElasticProperties {
    // Young's moduli in material principal directions [Pa]
    double E1, E2, E3;
    
    // Poisson's ratios [-]
    // nu_ij = -ε_j / ε_i under uniaxial stress σ_i
    // Symmetry requirement: nu_ij / E_i = nu_ji / E_j
    double nu12, nu13, nu23;
    
    // Shear moduli [Pa]
    double G12, G13, G23;
    
    // Density [kg/m³]
    double density;
    
    // Thermal expansion coefficients in each direction [1/K]
    double alpha1, alpha2, alpha3;
    
    // Material orientation: rotation matrix from global to material coords
    // Material coordinates: 1=fiber direction, 2=transverse, 3=through-thickness
    Tensor<2, 3> orientation_matrix;
    
    OrthotropicElasticProperties();
    
    // Build compliance tensor [S]: {ε} = [S]{σ}
    SymmetricTensor<4, 3> get_compliance_tensor() const;
    
    // Build stiffness tensor [C] = [S]^(-1): {σ} = [C]{ε}
    SymmetricTensor<4, 3> get_elasticity_tensor() const;
    
    // Get elasticity tensor rotated to global coordinates
    SymmetricTensor<4, 3> get_rotated_elasticity_tensor() const;
    
    // Get thermal expansion tensor (rotated to global)
    SymmetricTensor<2, 3> get_thermal_expansion_tensor() const;
    
    // Validate thermodynamic constraints
    // The compliance matrix must be positive definite
    bool is_thermodynamically_valid() const;
    
    // Set orientation from Euler angles (Z-X-Z convention, in radians)
    void set_orientation_euler(double phi, double theta, double psi);
    
    // Set orientation from a single fiber direction vector
    void set_orientation_from_fiber_direction(const Tensor<1, 3>& fiber_dir);
};

// ============================================================================
// Elastoplastic von Mises Properties
// ============================================================================

// Isotropic elasticity with J2 (von Mises) plasticity
// Supports isotropic, kinematic, or combined hardening
struct ElastoplasticVonMisesProperties {
    // Elastic properties
    double youngs_modulus;
    double poissons_ratio;
    double density;
    
    // Initial yield stress [Pa]
    double initial_yield_stress;
    
    // Hardening model
    enum class HardeningType { 
        PERFECT_PLASTIC,  // No hardening (elastic-perfectly plastic)
        LINEAR,           // σ_y = σ_y0 + H * ε_p^eq
        POWER_LAW,        // σ_y = σ_y0 * (1 + ε_p^eq / ε_0)^n
        TABULAR           // Piecewise linear from data points
    };
    HardeningType hardening_type = HardeningType::LINEAR;
    
    // For LINEAR hardening
    double isotropic_hardening_modulus;   // H [Pa] (slope of σ_y vs ε_p)
    double kinematic_hardening_modulus;   // C [Pa] (for back stress evolution)
    
    // For POWER_LAW hardening: σ_y = σ_y0 * (1 + ε_p / ε_0)^n
    double power_law_exponent;            // n [-]
    double power_law_reference_strain;    // ε_0 [-]
    
    // For TABULAR hardening: (equivalent plastic strain, yield stress) pairs
    std::vector<std::pair<double, double>> hardening_curve;
    
    double thermal_expansion_coeff;
    
    ElastoplasticVonMisesProperties();
    
    // Get current yield stress given equivalent plastic strain
    double get_yield_stress(double equiv_plastic_strain) const;
    
    // Get hardening modulus (dσ_y/dε_p) at given plastic strain
    double get_hardening_modulus(double equiv_plastic_strain) const;
    
    // Get elastic stiffness tensor
    SymmetricTensor<4, 3> get_elastic_tensor() const;
    
    // Get thermal expansion tensor
    SymmetricTensor<2, 3> get_thermal_expansion_tensor() const;
    
    bool is_valid() const;
};

// ============================================================================
// Hyperelastic Neo-Hookean Properties
// ============================================================================

// Compressible Neo-Hookean model for rubber-like materials
// Strain energy: W = (μ/2)(I1 - 3) - μ*ln(J) + (λ/2)(ln(J))²
// where I1 = tr(C), J = det(F), C = F^T * F
struct HyperelasticNeoHookeanProperties {
    double shear_modulus;     // μ [Pa] (initial shear modulus)
    double bulk_modulus;      // κ [Pa] (for volumetric response)
    double density;           // [kg/m³]
    
    HyperelasticNeoHookeanProperties()
        : shear_modulus(0), bulk_modulus(0), density(0) {}
    
    HyperelasticNeoHookeanProperties(double mu, double kappa, double rho)
        : shear_modulus(mu), bulk_modulus(kappa), density(rho) {}
    
    // Convert from E, nu (for user convenience)
    static HyperelasticNeoHookeanProperties from_engineering_constants(
        double E, double nu, double rho);
    
    // Compute 2nd Piola-Kirchhoff stress S from deformation gradient F
    // S = ∂W/∂E where E = 0.5*(C - I) is Green-Lagrange strain
    SymmetricTensor<2, 3> get_pk2_stress(
        const Tensor<2, 3>& deformation_gradient) const;
    
    // Compute material tangent ∂S/∂E (or equivalently 2*∂S/∂C)
    SymmetricTensor<4, 3> get_material_tangent(
        const Tensor<2, 3>& deformation_gradient) const;
    
    // Compute Cauchy stress σ from deformation gradient F
    // σ = (1/J) * F * S * F^T
    SymmetricTensor<2, 3> get_cauchy_stress(
        const Tensor<2, 3>& deformation_gradient) const;
    
    // Compute strain energy density
    double get_strain_energy(const Tensor<2, 3>& deformation_gradient) const;
    
    bool is_valid() const {
        return shear_modulus > 0 && bulk_modulus > 0 && density >= 0;
    }
};

// ============================================================================
// Hyperelastic Mooney-Rivlin Properties
// ============================================================================

// Two-parameter Mooney-Rivlin model (more accurate for rubber)
// Strain energy: W = C10*(I1-3) + C01*(I2-3) + (κ/2)*(J-1)²
// where I1, I2 are invariants of the deviatoric right Cauchy-Green tensor
struct HyperelasticMooneyRivlinProperties {
    double C10, C01;          // Material constants [Pa]
    double bulk_modulus;      // κ [Pa]
    double density;           // [kg/m³]
    
    HyperelasticMooneyRivlinProperties()
        : C10(0), C01(0), bulk_modulus(0), density(0) {}
    
    HyperelasticMooneyRivlinProperties(double c10, double c01, double kappa, double rho)
        : C10(c10), C01(c01), bulk_modulus(kappa), density(rho) {}
    
    // Initial shear modulus: μ = 2*(C10 + C01)
    double initial_shear_modulus() const {
        return 2.0 * (C10 + C01);
    }
    
    // Compute stresses and tangent (similar interface to Neo-Hookean)
    SymmetricTensor<2, 3> get_pk2_stress(
        const Tensor<2, 3>& deformation_gradient) const;
    
    SymmetricTensor<4, 3> get_material_tangent(
        const Tensor<2, 3>& deformation_gradient) const;
    
    SymmetricTensor<2, 3> get_cauchy_stress(
        const Tensor<2, 3>& deformation_gradient) const;
    
    // Compute strain energy density
    double get_strain_energy(const Tensor<2, 3>& deformation_gradient) const;
    
    bool is_valid() const {
        return (C10 + C01) > 0 && bulk_modulus > 0 && density >= 0;
    }
};

// ============================================================================
// Variant Type for All Material Properties
// ============================================================================

using MaterialProperties = std::variant<
    IsotropicElasticProperties,
    OrthotropicElasticProperties,
    ElastoplasticVonMisesProperties,
    HyperelasticNeoHookeanProperties,
    HyperelasticMooneyRivlinProperties
>;

// ============================================================================
// Complete Material Definition
// ============================================================================

struct Material {
    std::string id;                   // Unique identifier (e.g., "steel_a36")
    std::string name;                 // Display name (e.g., "Structural Steel ASTM A36")
    MaterialModel model;              // Which constitutive model
    MaterialProperties properties;    // Model-specific parameters
    
    // Design limits for safety factor calculations (optional)
    std::optional<double> yield_strength;      // [Pa]
    std::optional<double> ultimate_strength;   // [Pa]
    std::optional<double> fatigue_limit;       // [Pa] (endurance limit)
    
    // Check if this material requires nonlinear solution
    bool is_nonlinear() const {
        return model == MaterialModel::ELASTOPLASTIC_VONMISES ||
               model == MaterialModel::ELASTOPLASTIC_DRUCKER_PRAGER ||
               model == MaterialModel::HYPERELASTIC_NEOHOOKEAN ||
               model == MaterialModel::HYPERELASTIC_MOONEY_RIVLIN;
    }
    
    // Check if this material requires geometric nonlinearity
    bool requires_large_deformation() const {
        return model == MaterialModel::HYPERELASTIC_NEOHOOKEAN ||
               model == MaterialModel::HYPERELASTIC_MOONEY_RIVLIN;
    }
    
    // Get density from whatever property type is stored
    double get_density() const;
    
    // Get thermal expansion (isotropic average for anisotropic materials)
    double get_thermal_expansion() const;
    
    // Get elasticity tensor (for linear analysis or initial tangent)
    SymmetricTensor<4, 3> get_elasticity_tensor() const;
};

// ============================================================================
// Material Library Class
// ============================================================================

class MaterialLibrary {
public:
    MaterialLibrary();
    ~MaterialLibrary() = default;
    
    // Add a custom material
    void add_material(const Material& mat);
    
    // Get material by ID (throws if not found)
    const Material& get_material(const std::string& id) const;
    
    // Check if material exists
    bool has_material(const std::string& id) const;
    
    // List all material IDs
    std::vector<std::string> list_materials() const;
    
    // Get all materials (for API response)
    std::vector<Material> get_all_materials() const;
    
    // Remove a material
    void remove_material(const std::string& id);
    
    // Clear all materials (including presets)
    void clear();
    
    // Reload preset materials
    void reload_presets();
    
    // ===== Preset Material IDs =====
    static const std::string STEEL_STRUCTURAL;       // ASTM A36
    static const std::string STEEL_STAINLESS_304;    // 304 Stainless
    static const std::string STEEL_STAINLESS_316;    // 316 Stainless
    static const std::string STEEL_4340;             // AISI 4340
    static const std::string ALUMINUM_6061_T6;
    static const std::string ALUMINUM_7075_T6;
    static const std::string ALUMINUM_2024_T3;
    static const std::string TITANIUM_TI6AL4V;
    static const std::string COPPER_C10100;
    static const std::string BRASS_C36000;
    static const std::string CONCRETE_NORMAL;
    static const std::string CONCRETE_HIGH_STRENGTH;
    static const std::string CFRP_UNIDIRECTIONAL;    // Orthotropic
    static const std::string CFRP_WOVEN;             // Orthotropic
    static const std::string GFRP_UNIDIRECTIONAL;    // Orthotropic
    static const std::string GFRP_WOVEN;             // Orthotropic
    static const std::string WOOD_OAK;               // Orthotropic
    static const std::string WOOD_PINE;              // Orthotropic
    static const std::string RUBBER_NATURAL;         // Hyperelastic
    static const std::string RUBBER_NEOPRENE;        // Hyperelastic
    static const std::string SILICONE;               // Hyperelastic
    
private:
    std::map<std::string, Material> materials_;
    
    void populate_preset_materials();
    void add_steel_presets();
    void add_aluminum_presets();
    void add_other_metal_presets();
    void add_composite_presets();
    void add_rubber_presets();
    void add_other_presets();
};

// ============================================================================
// Helper Functions
// ============================================================================

// Compute von Mises equivalent stress from stress tensor
double compute_von_mises_stress(const SymmetricTensor<2, 3>& stress);

// Compute principal stresses (eigenvalues)
std::array<double, 3> compute_principal_stresses(const SymmetricTensor<2, 3>& stress);

// Compute hydrostatic (mean) stress
double compute_hydrostatic_stress(const SymmetricTensor<2, 3>& stress);

// Compute deviatoric stress tensor
SymmetricTensor<2, 3> compute_deviatoric_stress(const SymmetricTensor<2, 3>& stress);

} // namespace FEA

#endif // MATERIAL_LIBRARY_H
