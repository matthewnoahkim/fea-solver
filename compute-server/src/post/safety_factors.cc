#include "safety_factors.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace FEA {

// ============================================================================
// Constructor
// ============================================================================

template <int dim>
SafetyFactorCalculator<dim>::SafetyFactorCalculator(
    const DoFHandler<dim>& dh,
    const Mapping<dim>& map,
    const std::map<unsigned int, Material>& mats)
    : dof_handler_(dh)
    , mapping_(map)
    , materials_(mats)
    , current_criterion_(Criterion::VON_MISES_YIELD)
{}

// ============================================================================
// Configuration
// ============================================================================

template <int dim>
void SafetyFactorCalculator<dim>::set_custom_allowable(double allowable) {
    custom_allowable_ = allowable;
}

// ============================================================================
// Main Computation
// ============================================================================

template <int dim>
void SafetyFactorCalculator<dim>::compute(
    const Vector<double>& stress_field,
    Criterion criterion) {
    
    current_criterion_ = criterion;
    const unsigned int n_cells = stress_field.size();
    sf_field_.reinit(n_cells);
    
    stats_.min_sf = std::numeric_limits<double>::max();
    stats_.max_sf = 0;
    
    unsigned int cell_idx = 0;
    for (const auto& cell : dof_handler_.active_cell_iterators()) {
        double stress = stress_field(cell_idx);
        double allowable = get_allowable_stress(cell->material_id(), criterion);
        
        // Apply design factor (reduces allowable)
        allowable /= design_factor_;
        
        // Compute safety factor
        double sf;
        if (stress > 1e-10) {
            sf = allowable / stress;
        } else {
            sf = 1e6;  // Very high SF for near-zero stress
        }
        
        // Clamp to reasonable range for numerical stability
        sf = std::min(sf, 1e6);
        
        sf_field_(cell_idx) = sf;
        
        // Track minimum and maximum
        if (sf < stats_.min_sf) {
            stats_.min_sf = sf;
            stats_.min_sf_location = cell->center();
            stats_.min_sf_material_id = cell->material_id();
        }
        stats_.max_sf = std::max(stats_.max_sf, sf);
        
        ++cell_idx;
    }
    
    compute_statistics();
}

// ============================================================================
// Allowable Stress Lookup
// ============================================================================

template <int dim>
double SafetyFactorCalculator<dim>::get_allowable_stress(
    unsigned int material_id, Criterion criterion) const {
    
    // Custom allowable overrides material data
    if (custom_allowable_.has_value())
        return *custom_allowable_;
    
    // Find material
    auto it = materials_.find(material_id);
    if (it == materials_.end() && !materials_.empty())
        it = materials_.begin();
    
    if (it == materials_.end())
        return 1e12;  // Very high default (essentially infinite SF)
    
    const Material& mat = it->second;
    
    switch (criterion) {
        case Criterion::VON_MISES_YIELD:
        case Criterion::MAX_PRINCIPAL_YIELD:
            return mat.yield_strength.value_or(1e12);
            
        case Criterion::TRESCA_YIELD:
            // Tresca allowable = yield / 2 (for shear)
            return mat.yield_strength.value_or(1e12) / 2.0;
            
        case Criterion::VON_MISES_ULTIMATE:
        case Criterion::MAX_PRINCIPAL_ULTIMATE:
            return mat.ultimate_strength.value_or(1e12);
            
        case Criterion::TRESCA_ULTIMATE:
            return mat.ultimate_strength.value_or(1e12) / 2.0;
            
        case Criterion::GOODMAN_FATIGUE:
            // Use fatigue limit if available, otherwise estimate as 0.5 * Sut
            return mat.fatigue_limit.value_or(
                mat.ultimate_strength.value_or(1e12) * 0.5);
            
        case Criterion::CUSTOM_ALLOWABLE:
            return custom_allowable_.value_or(1e12);
    }
    
    return mat.yield_strength.value_or(1e12);
}

// ============================================================================
// Statistics Computation
// ============================================================================

template <int dim>
void SafetyFactorCalculator<dim>::compute_statistics() {
    if (sf_field_.size() == 0) return;
    
    double sum = 0;
    unsigned int below_1_0 = 0, below_1_25 = 0, below_1_5 = 0;
    unsigned int below_2_0 = 0, below_3_0 = 0;
    
    for (unsigned int i = 0; i < sf_field_.size(); ++i) {
        double sf = sf_field_(i);
        sum += sf;
        
        if (sf < 1.0) ++below_1_0;
        if (sf < 1.25) ++below_1_25;
        if (sf < 1.5) ++below_1_5;
        if (sf < 2.0) ++below_2_0;
        if (sf < 3.0) ++below_3_0;
    }
    
    unsigned int n = sf_field_.size();
    stats_.avg_sf = sum / n;
    stats_.percent_below_1_0 = 100.0 * below_1_0 / n;
    stats_.percent_below_1_25 = 100.0 * below_1_25 / n;
    stats_.percent_below_1_5 = 100.0 * below_1_5 / n;
    stats_.percent_below_2_0 = 100.0 * below_2_0 / n;
    stats_.percent_below_3_0 = 100.0 * below_3_0 / n;
    
    // Volume-weighted minimum (over worst 1% of elements)
    stats_.volume_weighted_min_sf = get_volume_weighted_min_sf(0.01);
}

template <int dim>
double SafetyFactorCalculator<dim>::get_percent_below_threshold(double threshold) const {
    unsigned int count = 0;
    for (unsigned int i = 0; i < sf_field_.size(); ++i) {
        if (sf_field_(i) < threshold)
            ++count;
    }
    return (sf_field_.size() > 0) ? 100.0 * count / sf_field_.size() : 0.0;
}

template <int dim>
double SafetyFactorCalculator<dim>::get_volume_weighted_min_sf(double volume_fraction) const {
    if (sf_field_.size() == 0) return 0;
    
    // Sort safety factors to find the worst percentile
    std::vector<double> sorted_sf(sf_field_.size());
    for (unsigned int i = 0; i < sf_field_.size(); ++i)
        sorted_sf[i] = sf_field_(i);
    
    std::sort(sorted_sf.begin(), sorted_sf.end());
    
    // Get the SF at the volume_fraction percentile
    unsigned int n_cells = std::max(1u,
        static_cast<unsigned int>(volume_fraction * sorted_sf.size()));
    
    return sorted_sf[n_cells - 1];
}

// ============================================================================
// Design Check
// ============================================================================

template <int dim>
bool SafetyFactorCalculator<dim>::passes_design_check(double required_sf) const {
    return stats_.min_sf >= required_sf;
}

// ============================================================================
// Report Generation
// ============================================================================

template <int dim>
std::string SafetyFactorCalculator<dim>::criterion_to_string(Criterion c) {
    switch (c) {
        case Criterion::VON_MISES_YIELD: return "von_mises_yield";
        case Criterion::VON_MISES_ULTIMATE: return "von_mises_ultimate";
        case Criterion::TRESCA_YIELD: return "tresca_yield";
        case Criterion::TRESCA_ULTIMATE: return "tresca_ultimate";
        case Criterion::MAX_PRINCIPAL_YIELD: return "max_principal_yield";
        case Criterion::MAX_PRINCIPAL_ULTIMATE: return "max_principal_ultimate";
        case Criterion::GOODMAN_FATIGUE: return "goodman_fatigue";
        case Criterion::CUSTOM_ALLOWABLE: return "custom_allowable";
    }
    return "unknown";
}

template <int dim>
std::string SafetyFactorCalculator<dim>::get_assessment_report() const {
    std::ostringstream report;
    report << std::fixed << std::setprecision(2);
    
    report << "\n";
    report << "╔══════════════════════════════════════════════════════════════╗\n";
    report << "║              SAFETY FACTOR ASSESSMENT                        ║\n";
    report << "╠══════════════════════════════════════════════════════════════╣\n";
    report << "║  Criterion: " << std::left << std::setw(48) 
           << criterion_to_string(current_criterion_) << "║\n";
    report << "╠══════════════════════════════════════════════════════════════╣\n";
    report << "║  Minimum Safety Factor: " << std::setw(10) << stats_.min_sf << "                      ║\n";
    report << "║  Location: (" << std::setw(8) << stats_.min_sf_location[0] << ", "
           << std::setw(8) << stats_.min_sf_location[1];
    if constexpr (dim == 3)
        report << ", " << std::setw(8) << stats_.min_sf_location[2];
    report << ")   ║\n";
    report << "║  Maximum Safety Factor: " << std::setw(10) << stats_.max_sf << "                      ║\n";
    report << "║  Average Safety Factor: " << std::setw(10) << stats_.avg_sf << "                      ║\n";
    report << "║  Volume-Weighted Min (1%): " << std::setw(8) << stats_.volume_weighted_min_sf 
           << "                    ║\n";
    report << "╠══════════════════════════════════════════════════════════════╣\n";
    report << "║  DISTRIBUTION:                                               ║\n";
    report << "║    Below 1.0 (FAIL):    " << std::setw(6) << stats_.percent_below_1_0 << "%                          ║\n";
    report << "║    Below 1.25:          " << std::setw(6) << stats_.percent_below_1_25 << "%                          ║\n";
    report << "║    Below 1.5:           " << std::setw(6) << stats_.percent_below_1_5 << "%                          ║\n";
    report << "║    Below 2.0:           " << std::setw(6) << stats_.percent_below_2_0 << "%                          ║\n";
    report << "║    Below 3.0:           " << std::setw(6) << stats_.percent_below_3_0 << "%                          ║\n";
    report << "╠══════════════════════════════════════════════════════════════╣\n";
    
    // Assessment
    std::string assessment;
    std::string symbol;
    if (stats_.min_sf >= 3.0) {
        assessment = "EXCELLENT - Generous safety margin";
        symbol = "✓✓";
    } else if (stats_.min_sf >= 2.0) {
        assessment = "PASS - Design has adequate safety margin";
        symbol = "✓";
    } else if (stats_.min_sf >= 1.5) {
        assessment = "MARGINAL - Review critical areas";
        symbol = "⚠";
    } else if (stats_.min_sf >= 1.0) {
        assessment = "WARNING - Low safety margin";
        symbol = "⚠⚠";
    } else {
        assessment = "FAIL - Design exceeds allowable stress";
        symbol = "✗";
    }
    
    report << "║  Assessment: " << symbol << " " << std::left << std::setw(46) 
           << assessment << "║\n";
    report << "╚══════════════════════════════════════════════════════════════╝\n";
    
    return report.str();
}

template <int dim>
json SafetyFactorCalculator<dim>::to_json() const {
    json j = stats_.to_json();
    j["criterion"] = criterion_to_string(current_criterion_);
    
    // Determine assessment string
    std::string assessment;
    if (stats_.min_sf >= 3.0) assessment = "EXCELLENT";
    else if (stats_.min_sf >= 2.0) assessment = "PASS";
    else if (stats_.min_sf >= 1.5) assessment = "MARGINAL";
    else if (stats_.min_sf >= 1.0) assessment = "WARNING";
    else assessment = "FAIL";
    
    j["assessment"] = assessment;
    
    return j;
}

// ============================================================================
// Explicit Instantiations
// ============================================================================

template class SafetyFactorCalculator<3>;
template class SafetyFactorCalculator<2>;

} // namespace FEA
