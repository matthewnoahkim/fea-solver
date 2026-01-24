#ifndef VTK_OUTPUT_H
#define VTK_OUTPUT_H

/**
 * @file vtk_output.h
 * @brief VTK/VTU output for visualization of FEA results
 * 
 * Generates visualization files compatible with ParaView:
 * - VTK (ASCII) format
 * - VTU (XML binary) format
 * - PVD collection files for time series
 */

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping.h>

#include "../solver/material_library.h"

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace FEA {

using namespace dealii;

/**
 * @brief VTK/VTU output manager for FEA results
 * 
 * @tparam dim Spatial dimension (2 or 3)
 */
template <int dim>
class VTKOutput {
public:
    /**
     * @brief Output format options
     */
    enum class Format {
        VTK_ASCII,      ///< Legacy VTK ASCII format
        VTK_BINARY,     ///< Legacy VTK binary format
        VTU_ASCII,      ///< Modern XML ASCII format
        VTU_BINARY,     ///< Modern XML binary format (default)
        VTU_COMPRESSED  ///< VTU with zlib compression
    };
    
    /**
     * @brief Output configuration
     */
    struct OutputConfig {
        Format format = Format::VTU_BINARY;
        unsigned int subdivisions = 0;  ///< Additional subdivisions for higher-order elements
        bool output_partitioning = false;  ///< Include MPI partitioning info
        bool output_material_ids = true;   ///< Include material IDs
        bool output_cell_data = true;      ///< Include cell-averaged data
        bool output_point_data = true;     ///< Include nodal data
    };
    
    /**
     * @brief Construct output manager
     */
    VTKOutput(const DoFHandler<dim>& dof_handler,
              const Mapping<dim>& mapping);
    
    /**
     * @brief Set output configuration
     */
    void set_config(const OutputConfig& config) { config_ = config; }
    
    // =========================================================================
    // Add Data Fields
    // =========================================================================
    
    /**
     * @brief Add displacement solution
     */
    void add_displacement(const Vector<double>& solution,
                          const std::string& name = "displacement");
    
    /**
     * @brief Add scalar field (cell data)
     */
    void add_cell_scalar(const Vector<double>& data,
                         const std::string& name);
    
    /**
     * @brief Add scalar field (point data)
     */
    void add_point_scalar(const Vector<double>& data,
                          const std::string& name);
    
    /**
     * @brief Add vector field (cell data)
     */
    void add_cell_vector(const std::vector<Tensor<1, dim>>& data,
                         const std::string& name);
    
    /**
     * @brief Add stress postprocessor (computes at nodes)
     */
    void add_stress_output(const std::map<unsigned int, Material>& materials);
    
    /**
     * @brief Add von Mises stress postprocessor
     */
    void add_von_mises_output(const std::map<unsigned int, Material>& materials);
    
    // =========================================================================
    // Output
    // =========================================================================
    
    /**
     * @brief Write output to file
     * @param filename Base filename (extension added automatically)
     * @return Full path of written file
     */
    std::string write(const std::string& filename);
    
    /**
     * @brief Write output with timestep (for transient/nonlinear)
     * @param base_filename Base filename
     * @param timestep Time step number
     * @param time Simulation time
     */
    void write_timestep(const std::string& base_filename,
                        unsigned int timestep,
                        double time);
    
    /**
     * @brief Write PVD collection file for time series
     * @param filename PVD filename
     */
    void write_pvd(const std::string& filename);
    
    /**
     * @brief Clear all added data (prepare for new output)
     */
    void clear();
    
private:
    const DoFHandler<dim>& dof_handler_;
    const Mapping<dim>& mapping_;
    OutputConfig config_;
    
    std::unique_ptr<DataOut<dim>> data_out_;
    
    // Time series tracking
    std::vector<std::pair<double, std::string>> pvd_entries_;
    
    void setup_data_out();
    std::string get_extension() const;
};

} // namespace FEA

#endif // VTK_OUTPUT_H
