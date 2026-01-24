#include "vtk_output.h"
#include "stress_calculator.h"
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/data_out_base.h>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace FEA {

template <int dim>
VTKOutput<dim>::VTKOutput(
    const DoFHandler<dim>& dh,
    const Mapping<dim>& map)
    : dof_handler_(dh)
    , mapping_(map)
{
    setup_data_out();
}

template <int dim>
void VTKOutput<dim>::setup_data_out() {
    data_out_ = std::make_unique<DataOut<dim>>();
    data_out_->attach_dof_handler(dof_handler_);
}

template <int dim>
std::string VTKOutput<dim>::get_extension() const {
    switch (config_.format) {
        case Format::VTK_ASCII:
        case Format::VTK_BINARY:
            return ".vtk";
        case Format::VTU_ASCII:
        case Format::VTU_BINARY:
        case Format::VTU_COMPRESSED:
        default:
            return ".vtu";
    }
}

template <int dim>
void VTKOutput<dim>::add_displacement(
    const Vector<double>& solution,
    const std::string& name) {
    
    std::vector<std::string> component_names;
    for (unsigned int d = 0; d < dim; ++d) {
        component_names.push_back(name + "_" + std::to_string(d));
    }
    
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    
    data_out_->add_data_vector(solution, component_names,
                               DataOut<dim>::type_dof_data,
                               component_interpretation);
}

template <int dim>
void VTKOutput<dim>::add_cell_scalar(
    const Vector<double>& data,
    const std::string& name) {
    
    data_out_->add_data_vector(data, name, DataOut<dim>::type_cell_data);
}

template <int dim>
void VTKOutput<dim>::add_point_scalar(
    const Vector<double>& data,
    const std::string& name) {
    
    data_out_->add_data_vector(data, name, DataOut<dim>::type_dof_data);
}

template <int dim>
void VTKOutput<dim>::add_cell_vector(
    const std::vector<Tensor<1, dim>>& data,
    const std::string& name) {
    
    // Convert to component-wise vectors
    Vector<double> vec_x(data.size()), vec_y(data.size());
    Vector<double> vec_z(data.size());
    
    for (size_t i = 0; i < data.size(); ++i) {
        vec_x(i) = data[i][0];
        vec_y(i) = data[i][1];
        if constexpr (dim == 3) {
            vec_z(i) = data[i][2];
        }
    }
    
    data_out_->add_data_vector(vec_x, name + "_x", DataOut<dim>::type_cell_data);
    data_out_->add_data_vector(vec_y, name + "_y", DataOut<dim>::type_cell_data);
    if constexpr (dim == 3) {
        data_out_->add_data_vector(vec_z, name + "_z", DataOut<dim>::type_cell_data);
    }
}

template <int dim>
void VTKOutput<dim>::add_stress_output(
    const std::map<unsigned int, Material>& materials) {
    
    auto stress_pp = std::make_shared<StressPostprocessor<dim>>(materials);
    data_out_->add_data_vector(dof_handler_.get_fe().n_components(),
                               *stress_pp);
}

template <int dim>
void VTKOutput<dim>::add_von_mises_output(
    const std::map<unsigned int, Material>& materials) {
    
    auto vm_pp = std::make_shared<VonMisesPostprocessor<dim>>(materials);
    data_out_->add_data_vector(dof_handler_.get_fe().n_components(),
                               *vm_pp);
}

template <int dim>
std::string VTKOutput<dim>::write(const std::string& filename) {
    // Build output
    if (config_.subdivisions > 0) {
        data_out_->build_patches(mapping_, config_.subdivisions);
    } else {
        data_out_->build_patches(mapping_);
    }
    
    std::string full_filename = filename + get_extension();
    std::ofstream output(full_filename);
    
    DataOutBase::VtkFlags vtk_flags;
    
    switch (config_.format) {
        case Format::VTK_ASCII:
            data_out_->write_vtk(output);
            break;
            
        case Format::VTK_BINARY:
            vtk_flags.write_higher_order_cells = false;
            data_out_->set_flags(vtk_flags);
            data_out_->write_vtk(output);
            break;
            
        case Format::VTU_ASCII:
            vtk_flags.write_higher_order_cells = true;
            data_out_->set_flags(vtk_flags);
            data_out_->write_vtu(output);
            break;
            
        case Format::VTU_BINARY:
            vtk_flags.write_higher_order_cells = true;
            data_out_->set_flags(vtk_flags);
            data_out_->write_vtu(output);
            break;
            
        case Format::VTU_COMPRESSED:
#ifdef DEAL_II_WITH_ZLIB
            vtk_flags.compression_level = DataOutBase::CompressionLevel::best_compression;
#endif
            vtk_flags.write_higher_order_cells = true;
            data_out_->set_flags(vtk_flags);
            data_out_->write_vtu(output);
            break;
    }
    
    return full_filename;
}

template <int dim>
void VTKOutput<dim>::write_timestep(
    const std::string& base_filename,
    unsigned int timestep,
    double time) {
    
    std::ostringstream filename;
    filename << base_filename << "_" << std::setfill('0') << std::setw(6) << timestep;
    
    std::string written_file = write(filename.str());
    pvd_entries_.emplace_back(time, written_file);
}

template <int dim>
void VTKOutput<dim>::write_pvd(const std::string& filename) {
    std::ofstream pvd_output(filename);
    
    pvd_output << "<?xml version=\"1.0\"?>\n";
    pvd_output << "<VTKFile type=\"Collection\" version=\"0.1\">\n";
    pvd_output << "  <Collection>\n";
    
    for (const auto& [time, file] : pvd_entries_) {
        pvd_output << "    <DataSet timestep=\"" << time 
                   << "\" file=\"" << file << "\"/>\n";
    }
    
    pvd_output << "  </Collection>\n";
    pvd_output << "</VTKFile>\n";
}

template <int dim>
void VTKOutput<dim>::clear() {
    data_out_.reset();
    setup_data_out();
}

// Explicit instantiation
template class VTKOutput<3>;
template class VTKOutput<2>;

} // namespace FEA
