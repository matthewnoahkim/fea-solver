#include "linearized_stress.h"
#include "stress_calculator.h"
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/fe_values.h>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace FEA {

// ============================================================================
// Constructor
// ============================================================================

template <int dim>
LinearizedStressCalculator<dim>::LinearizedStressCalculator(
    const DoFHandler<dim>& dh,
    const Mapping<dim>& map,
    const std::map<unsigned int, Material>& mats)
    : dof_handler_(dh)
    , mapping_(map)
    , materials_(mats)
{}

// ============================================================================
// SCL Management
// ============================================================================

template <int dim>
void LinearizedStressCalculator<dim>::add_scl(
    const Point<dim>& start,
    const Point<dim>& end,
    const std::string& name,
    unsigned int num_points) {
    
    StressClassificationLine scl;
    scl.start = start;
    scl.end = end;
    scl.name = name.empty() ? "SCL_" + std::to_string(scls_.size() + 1) : name;
    scl.num_points = num_points;
    scls_.push_back(scl);
}

template <int dim>
void LinearizedStressCalculator<dim>::add_scl(const StressClassificationLine& scl) {
    scls_.push_back(scl);
}

template <int dim>
void LinearizedStressCalculator<dim>::add_scls(
    const std::vector<StressClassificationLine>& scls) {
    for (const auto& scl : scls)
        scls_.push_back(scl);
}

template <int dim>
void LinearizedStressCalculator<dim>::clear_scls() {
    scls_.clear();
    results_.clear();
}

// ============================================================================
// Main Computation
// ============================================================================

template <int dim>
void LinearizedStressCalculator<dim>::compute(const Vector<double>& solution) {
    results_.clear();
    results_.reserve(scls_.size());
    
    for (const auto& scl : scls_) {
        compute_scl(scl, solution);
    }
}

template <int dim>
void LinearizedStressCalculator<dim>::compute_scl(
    const StressClassificationLine& scl,
    const Vector<double>& solution) {
    
    LinearizedResult result;
    result.scl_name = scl.name;
    result.start = scl.start;
    result.end = scl.end;
    
    // Compute SCL thickness (length)
    double thickness = scl.start.distance(scl.end);
    result.thickness = thickness;
    
    if (thickness < 1e-10) {
        results_.push_back(result);
        return;
    }
    
    // Direction vector along SCL (from inner to outer surface)
    Tensor<1, dim> direction;
    for (unsigned int d = 0; d < dim; ++d)
        direction[d] = (scl.end[d] - scl.start[d]) / thickness;
    
    // Sample stress at points along the SCL
    std::vector<SymmetricTensor<2, dim>> stress_samples(scl.num_points);
    std::vector<double> positions(scl.num_points);  // Position from midpoint
    
    for (unsigned int i = 0; i < scl.num_points; ++i) {
        // Parametric position [0, 1]
        double t = static_cast<double>(i) / (scl.num_points - 1);
        
        // Position from midpoint [-t/2, t/2]
        positions[i] = t * thickness - thickness / 2.0;
        
        // Physical sample point
        Point<dim> sample_point;
        for (unsigned int d = 0; d < dim; ++d)
            sample_point[d] = scl.start[d] + t * (scl.end[d] - scl.start[d]);
        
        stress_samples[i] = get_stress_at_point(sample_point, solution);
    }
    
    // =========================================================================
    // Stress Linearization
    // =========================================================================
    // Per ASME BPVC Section VIII Division 2, Annex 5-A:
    //
    // Membrane stress: σ_m = (1/t) ∫ σ(x) dx
    // Bending stress:  σ_b = (6/t²) ∫ σ(x) * x dx
    //   where x is measured from the mid-surface
    //
    // Peak stress: σ_f = σ_total - (σ_m + σ_b)
    
    SymmetricTensor<2, dim> membrane_stress;
    SymmetricTensor<2, dim> bending_stress;
    
    // Numerical integration using trapezoidal rule
    double dt = thickness / (scl.num_points - 1);
    
    for (unsigned int i = 0; i < scl.num_points; ++i) {
        // Trapezoidal weights
        double weight = (i == 0 || i == scl.num_points - 1) ? 0.5 : 1.0;
        weight *= dt / thickness;  // Normalize by thickness
        
        // Membrane: average stress
        membrane_stress += weight * stress_samples[i];
        
        // Bending: (6/t²) * ∫ σ*x dx = (6/t) * ∫ σ*(x/t) d(x/t)
        bending_stress += weight * positions[i] * stress_samples[i] * (6.0 / thickness);
    }
    
    result.membrane_tensor = membrane_stress;
    result.bending_tensor = bending_stress;
    
    // =========================================================================
    // Equivalent Stresses
    // =========================================================================
    
    result.membrane = compute_von_mises(membrane_stress);
    result.bending = compute_von_mises(bending_stress);
    
    // Membrane + Bending combined
    SymmetricTensor<2, dim> mpb_stress = membrane_stress + bending_stress;
    result.membrane_plus_bending = compute_von_mises(mpb_stress);
    
    // Peak stress at surface
    // Linearized stress at inner surface: σ_m - σ_b (tension on outside)
    // Linearized stress at outer surface: σ_m + σ_b
    // Peak = total - linearized
    
    // Check both surfaces, use maximum peak
    SymmetricTensor<2, dim> surface_inner = stress_samples[0];
    SymmetricTensor<2, dim> surface_outer = stress_samples[scl.num_points - 1];
    
    SymmetricTensor<2, dim> linearized_inner = membrane_stress - bending_stress;
    SymmetricTensor<2, dim> linearized_outer = membrane_stress + bending_stress;
    
    SymmetricTensor<2, dim> peak_inner = surface_inner - linearized_inner;
    SymmetricTensor<2, dim> peak_outer = surface_outer - linearized_outer;
    
    double peak_vm_inner = compute_von_mises(peak_inner);
    double peak_vm_outer = compute_von_mises(peak_outer);
    result.peak = std::max(peak_vm_inner, peak_vm_outer);
    
    // Total stress (maximum of inner/outer surfaces)
    double total_inner = compute_von_mises(surface_inner);
    double total_outer = compute_von_mises(surface_outer);
    result.total = std::max(total_inner, total_outer);
    
    // =========================================================================
    // Stress Intensity (ASME uses Tresca-based SI)
    // =========================================================================
    
    result.membrane_intensity = compute_stress_intensity(membrane_stress);
    result.membrane_plus_bending_intensity = compute_stress_intensity(mpb_stress);
    
    // =========================================================================
    // ASME Allowable Comparison
    // =========================================================================
    
    // Get Sm from material (use first material if unknown)
    result.Sm = get_sm_allowable(0);
    
    if (result.Sm > 0) {
        // Membrane: Pm <= Sm
        result.membrane_utilization = result.membrane_intensity / result.Sm;
        result.membrane_ok = result.membrane_intensity <= result.Sm;
        
        // Membrane + Bending: Pm + Pb <= 1.5 * Sm
        result.membrane_plus_bending_utilization = 
            result.membrane_plus_bending_intensity / (1.5 * result.Sm);
        result.membrane_plus_bending_ok = 
            result.membrane_plus_bending_intensity <= 1.5 * result.Sm;
    } else {
        result.membrane_utilization = 0;
        result.membrane_plus_bending_utilization = 0;
        result.membrane_ok = true;
        result.membrane_plus_bending_ok = true;
    }
    
    results_.push_back(result);
}

// ============================================================================
// Stress at Point
// ============================================================================

template <int dim>
SymmetricTensor<2, dim> LinearizedStressCalculator<dim>::get_stress_at_point(
    const Point<dim>& p,
    const Vector<double>& solution) const {
    
    try {
        auto cell_and_point = GridTools::find_active_cell_around_point(
            mapping_, dof_handler_, p);
        
        auto cell = cell_and_point.first;
        auto ref_point = cell_and_point.second;
        
        if (cell == dof_handler_.end())
            return SymmetricTensor<2, dim>();
        
        const auto& fe = dof_handler_.get_fe();
        Quadrature<dim> point_quadrature(ref_point);
        FEValues<dim> fe_values(mapping_, fe, point_quadrature,
            update_values | update_gradients);
        
        fe_values.reinit(cell);
        
        std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
        cell->get_dof_indices(dof_indices);
        
        // Compute strain from displacement gradients
        SymmetricTensor<2, dim> strain;
        for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i) {
            unsigned int comp = fe.system_to_component_index(i).first;
            double u_i = solution(dof_indices[i]);
            const Tensor<1, dim>& grad = fe_values.shape_grad(i, 0);
            
            for (unsigned int d = 0; d < dim; ++d) {
                strain[comp][d] += 0.5 * u_i * grad[d];
                strain[d][comp] += 0.5 * u_i * grad[d];
            }
        }
        
        // Get material and compute stress σ = C : ε
        auto mat_it = materials_.find(cell->material_id());
        if (mat_it == materials_.end() && !materials_.empty())
            mat_it = materials_.begin();
        
        if (mat_it != materials_.end()) {
            if (auto* iso = std::get_if<IsotropicElasticProperties>(&mat_it->second.properties)) {
                return iso->get_elasticity_tensor() * strain;
            }
            else if (auto* ortho = std::get_if<OrthotropicElasticProperties>(&mat_it->second.properties)) {
                return ortho->get_elasticity_tensor() * strain;
            }
        }
        
        // Default: return strain (effectively E=1, nu=0)
        return SymmetricTensor<2, dim>();
        
    } catch (...) {
        // Point outside mesh
        return SymmetricTensor<2, dim>();
    }
}

// ============================================================================
// Stress Measures
// ============================================================================

template <int dim>
double LinearizedStressCalculator<dim>::compute_stress_intensity(
    const SymmetricTensor<2, dim>& stress) const {
    
    // ASME stress intensity = maximum principal stress difference
    // SI = max(|σ1 - σ2|, |σ2 - σ3|, |σ3 - σ1|) = σ1 - σ3
    
    auto eigen = eigenvectors(stress);
    std::array<double, dim> principals;
    for (unsigned int d = 0; d < dim; ++d)
        principals[d] = eigen[d].first;
    
    std::sort(principals.begin(), principals.end(), std::greater<double>());
    
    // For 3D: SI = σ1 - σ3
    // For 2D: SI = σ1 - σ2 (assuming plane stress with σ3 = 0)
    if constexpr (dim == 3) {
        return principals[0] - principals[2];
    } else {
        // For plane stress, need to consider σ3 = 0
        double s1 = principals[0];
        double s2 = principals[1];
        double s3 = 0.0;  // Plane stress assumption
        
        return std::max({std::abs(s1 - s2), std::abs(s2 - s3), std::abs(s3 - s1)});
    }
}

template <int dim>
double LinearizedStressCalculator<dim>::compute_von_mises(
    const SymmetricTensor<2, dim>& stress) const {
    
    double trace = 0;
    for (unsigned int d = 0; d < dim; ++d)
        trace += stress[d][d];
    
    double mean = trace / 3.0;
    
    SymmetricTensor<2, dim> dev = stress;
    for (unsigned int d = 0; d < dim; ++d)
        dev[d][d] -= mean;
    
    return std::sqrt(1.5 * (dev * dev));
}

template <int dim>
double LinearizedStressCalculator<dim>::get_sm_allowable(unsigned int material_id) const {
    auto it = materials_.find(material_id);
    if (it == materials_.end() && !materials_.empty())
        it = materials_.begin();
    
    if (it == materials_.end())
        return 0;
    
    const Material& mat = it->second;
    
    // ASME Sm is typically:
    // - Sy / 1.5 for yield
    // - Su / 2.4 for ultimate (Division 2)
    // Use the lower of the two
    
    double sm_yield = mat.yield_strength.has_value() ? 
                      *mat.yield_strength / 1.5 : 1e12;
    double sm_ult = mat.ultimate_strength.has_value() ?
                    *mat.ultimate_strength / 2.4 : 1e12;
    
    return std::min(sm_yield, sm_ult);
}

// ============================================================================
// Results
// ============================================================================

template <int dim>
typename LinearizedStressCalculator<dim>::LinearizedResult
LinearizedStressCalculator<dim>::get_result(const std::string& name) const {
    for (const auto& r : results_) {
        if (r.scl_name == name)
            return r;
    }
    return LinearizedResult{};
}

template <int dim>
bool LinearizedStressCalculator<dim>::all_pass() const {
    for (const auto& r : results_) {
        if (!r.membrane_ok || !r.membrane_plus_bending_ok)
            return false;
    }
    return true;
}

// ============================================================================
// Report Generation
// ============================================================================

template <int dim>
std::string LinearizedStressCalculator<dim>::get_report() const {
    std::ostringstream report;
    report << std::fixed << std::setprecision(2);
    
    report << "\n";
    report << "╔══════════════════════════════════════════════════════════════╗\n";
    report << "║      LINEARIZED STRESS REPORT (ASME BPVC Div. 2)             ║\n";
    report << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    for (const auto& r : results_) {
        report << "┌──────────────────────────────────────────────────────────────┐\n";
        report << "│ SCL: " << std::left << std::setw(55) << r.scl_name << "│\n";
        report << "├──────────────────────────────────────────────────────────────┤\n";
        report << "│  Thickness: " << std::setw(10) << r.thickness << " m                                  │\n";
        report << "├──────────────────────────────────────────────────────────────┤\n";
        report << "│  Stress Components (von Mises equivalent):                   │\n";
        report << "│    Membrane (Pm):       " << std::setw(12) << r.membrane << " Pa                 │\n";
        report << "│    Bending (Pb):        " << std::setw(12) << r.bending << " Pa                 │\n";
        report << "│    Peak (F):            " << std::setw(12) << r.peak << " Pa                 │\n";
        report << "│    Pm + Pb:             " << std::setw(12) << r.membrane_plus_bending << " Pa                 │\n";
        report << "│    Total:               " << std::setw(12) << r.total << " Pa                 │\n";
        report << "├──────────────────────────────────────────────────────────────┤\n";
        report << "│  Stress Intensity (Tresca):                                  │\n";
        report << "│    Membrane SI:         " << std::setw(12) << r.membrane_intensity << " Pa                 │\n";
        report << "│    (Pm+Pb) SI:          " << std::setw(12) << r.membrane_plus_bending_intensity 
               << " Pa                 │\n";
        
        if (r.Sm > 0) {
            report << "├──────────────────────────────────────────────────────────────┤\n";
            report << "│  ASME Allowable Check:                                       │\n";
            report << "│    Sm (allowable):      " << std::setw(12) << r.Sm << " Pa                 │\n";
            report << "│    Pm/Sm:               " << std::setw(8) << r.membrane_utilization 
                   << (r.membrane_ok ? "    [OK]    " : " [EXCEEDS]  ") << "           │\n";
            report << "│    (Pm+Pb)/(1.5*Sm):    " << std::setw(8) << r.membrane_plus_bending_utilization
                   << (r.membrane_plus_bending_ok ? "    [OK]    " : " [EXCEEDS]  ") << "           │\n";
        }
        report << "└──────────────────────────────────────────────────────────────┘\n\n";
    }
    
    // Overall summary
    bool all_ok = all_pass();
    report << "Overall Assessment: " << (all_ok ? "✓ ALL SCLs PASS" : "✗ SOME SCLs EXCEED LIMITS") << "\n";
    
    return report.str();
}

template <int dim>
json LinearizedStressCalculator<dim>::to_json() const {
    json j = json::array();
    for (const auto& r : results_) {
        j.push_back(r.to_json());
    }
    return j;
}

// ============================================================================
// Explicit Instantiations
// ============================================================================

template class LinearizedStressCalculator<3>;
template class LinearizedStressCalculator<2>;

} // namespace FEA
