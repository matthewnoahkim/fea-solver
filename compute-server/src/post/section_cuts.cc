#include "section_cuts.h"
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace FEA {

template <int dim>
SectionCutAnalyzer<dim>::SectionCutAnalyzer(
    const DoFHandler<dim>& dh,
    const Mapping<dim>& map,
    const std::map<unsigned int, Material>& mats)
    : dof_handler_(dh)
    , mapping_(map)
    , materials_(mats)
{}

template <int dim>
void SectionCutAnalyzer<dim>::add_cut(
    const Point<dim>& point,
    const Tensor<1, dim>& normal,
    const std::string& name) {
    
    SectionCut cut;
    cut.point = point;
    cut.normal = normal / normal.norm();  // Normalize
    cut.name = name.empty() ? "Cut_" + std::to_string(cuts_.size() + 1) : name;
    cuts_.push_back(cut);
}

template <int dim>
void SectionCutAnalyzer<dim>::add_cut(const SectionCut& cut) {
    cuts_.push_back(cut);
}

template <int dim>
void SectionCutAnalyzer<dim>::clear_cuts() {
    cuts_.clear();
    results_.clear();
}

template <int dim>
void SectionCutAnalyzer<dim>::compute(const Vector<double>& solution) {
    results_.clear();
    results_.reserve(cuts_.size());
    
    for (const auto& cut : cuts_) {
        compute_cut(cut, solution);
    }
}

template <int dim>
void SectionCutAnalyzer<dim>::compute_cut(
    const SectionCut& cut,
    const Vector<double>& solution) {
    
    SectionResult result;
    result.name = cut.name;
    result.force = Tensor<1, dim>();
    result.moment = Tensor<1, dim>();
    result.area = 0;
    result.centroid = Point<dim>();
    
    const auto& fe = dof_handler_.get_fe();
    QGauss<dim> quadrature(fe.degree + 1);
    FEValues<dim> fe_values(mapping_, fe, quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);
    
    std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
    
    // Accumulate area-weighted centroid
    Point<dim> centroid_sum;
    double area_sum = 0;
    
    // Iterate through cells that intersect the cutting plane
    for (const auto& cell : dof_handler_.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell->get_dof_indices(dof_indices);
        
        // Check if cell intersects the cutting plane
        int pos_count = 0, neg_count = 0;
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
            Tensor<1, dim> r;
            for (unsigned int d = 0; d < dim; ++d)
                r[d] = cell->vertex(v)[d] - cut.point[d];
            
            double dist = r * cut.normal;
            if (dist > 1e-10) ++pos_count;
            else if (dist < -1e-10) ++neg_count;
        }
        
        // Skip cells that don't cross the plane
        if (pos_count == 0 || neg_count == 0) continue;
        
        // Get material elasticity
        SymmetricTensor<4, dim> C;
        auto mat_it = materials_.find(cell->material_id());
        if (mat_it == materials_.end() && !materials_.empty())
            mat_it = materials_.begin();
        
        if (mat_it != materials_.end()) {
            if (auto* iso = std::get_if<IsotropicElasticProperties>(&mat_it->second.properties)) {
                C = iso->get_elasticity_tensor();
            }
        }
        
        // Integrate stress over quadrature points near the plane
        for (unsigned int q = 0; q < quadrature.size(); ++q) {
            const Point<dim>& qp = fe_values.quadrature_point(q);
            
            // Check if this quadrature point is near the cutting plane
            Tensor<1, dim> r;
            for (unsigned int d = 0; d < dim; ++d)
                r[d] = qp[d] - cut.point[d];
            
            double dist = std::abs(r * cut.normal);
            double cell_size = cell->diameter();
            
            // Only include points very close to the plane
            if (dist > 0.1 * cell_size) continue;
            
            // Compute strain
            SymmetricTensor<2, dim> strain;
            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i) {
                unsigned int comp = fe.system_to_component_index(i).first;
                double u_i = solution(dof_indices[i]);
                const Tensor<1, dim>& grad = fe_values.shape_grad(i, q);
                
                for (unsigned int d = 0; d < dim; ++d) {
                    strain[comp][d] += 0.5 * u_i * grad[d];
                    strain[d][comp] += 0.5 * u_i * grad[d];
                }
            }
            
            // Compute stress
            SymmetricTensor<2, dim> stress = C * strain;
            
            // Compute traction on the cutting plane: t = σ · n
            Tensor<1, dim> traction;
            for (unsigned int i = 0; i < dim; ++i) {
                for (unsigned int j = 0; j < dim; ++j) {
                    traction[i] += stress[i][j] * cut.normal[j];
                }
            }
            
            double JxW = fe_values.JxW(q);
            
            // Accumulate force
            result.force += traction * JxW;
            
            // Accumulate moment about cut point
            if constexpr (dim == 3) {
                Tensor<1, dim> arm;
                for (unsigned int d = 0; d < dim; ++d)
                    arm[d] = qp[d] - cut.point[d];
                result.moment += cross_product_3d(arm, traction) * JxW;
            }
            
            // Accumulate area and centroid
            area_sum += JxW;
            for (unsigned int d = 0; d < dim; ++d)
                centroid_sum[d] += qp[d] * JxW;
        }
    }
    
    // Finalize centroid
    if (area_sum > 1e-14) {
        for (unsigned int d = 0; d < dim; ++d)
            result.centroid[d] = centroid_sum[d] / area_sum;
    }
    result.area = area_sum;
    
    // Compute local force components
    result.axial_force = result.force * cut.normal;
    
    // Shear is force component perpendicular to normal
    Tensor<1, dim> shear_vec = result.force - result.axial_force * cut.normal;
    if constexpr (dim == 3) {
        result.shear_force[0] = shear_vec[0];  // Simplified - would need local axes
        result.shear_force[1] = shear_vec[1];
        result.bending_moment_y = result.moment[1];
        result.bending_moment_z = result.moment[2];
        result.torsion = result.moment * cut.normal;
    } else {
        result.shear_force[0] = shear_vec.norm();
        result.bending_moment_y = result.moment[0];  // 2D moment is scalar
    }
    
    results_.push_back(result);
}

template <int dim>
typename SectionCutAnalyzer<dim>::SectionResult
SectionCutAnalyzer<dim>::get_result(const std::string& name) const {
    for (const auto& r : results_) {
        if (r.name == name)
            return r;
    }
    return SectionResult{};
}

template <int dim>
std::string SectionCutAnalyzer<dim>::get_report() const {
    std::ostringstream report;
    report << std::fixed << std::setprecision(4);
    
    report << "\n=== SECTION CUT RESULTS ===\n\n";
    
    for (const auto& r : results_) {
        report << "Cut: " << r.name << "\n";
        report << "  Area: " << r.area << " m²\n";
        report << "  Force: [" << r.force[0] << ", " << r.force[1];
        if constexpr (dim == 3) report << ", " << r.force[2];
        report << "] N\n";
        report << "  Axial: " << r.axial_force << " N\n";
        report << "  Moment: [" << r.moment[0] << ", " << r.moment[1];
        if constexpr (dim == 3) report << ", " << r.moment[2];
        report << "] N·m\n\n";
    }
    
    return report.str();
}

template <int dim>
json SectionCutAnalyzer<dim>::to_json() const {
    json j = json::array();
    for (const auto& r : results_)
        j.push_back(r.to_json());
    return j;
}

// Explicit instantiation
template class SectionCutAnalyzer<3>;
template class SectionCutAnalyzer<2>;

} // namespace FEA
