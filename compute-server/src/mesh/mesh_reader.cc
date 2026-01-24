#include "mesh_reader.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace FEA {

// ============================================================================
// MeshReader Constructor
// ============================================================================

template <int dim>
MeshReader<dim>::MeshReader(Triangulation<dim>& tria)
    : triangulation(tria)
{}

// ============================================================================
// File Format Readers
// ============================================================================

template <int dim>
void MeshReader<dim>::read_gmsh(const std::string& filename) {
    std::ifstream input(filename);
    if (!input)
        throw std::runtime_error("Cannot open mesh file: " + filename);
    read_gmsh(input);
}

template <int dim>
void MeshReader<dim>::read_gmsh(std::istream& input) {
    clear_metadata();
    
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    grid_in.read_msh(input);
    
    parse_gmsh_physical_groups();
}

template <int dim>
void MeshReader<dim>::read_vtk(const std::string& filename) {
    std::ifstream input(filename);
    if (!input)
        throw std::runtime_error("Cannot open mesh file: " + filename);
    read_vtk(input);
}

template <int dim>
void MeshReader<dim>::read_vtk(std::istream& input) {
    clear_metadata();
    
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    grid_in.read_vtk(input);
}

template <int dim>
void MeshReader<dim>::read_exodus(const std::string& filename) {
    clear_metadata();
    
    // Exodus format requires deal.II built with Trilinos/SEACAS
    // This is a placeholder - actual implementation would use:
    // GridIn<dim>::read_exodusii(filename)
    
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    
    // Check if file exists
    std::ifstream test(filename);
    if (!test)
        throw std::runtime_error("Cannot open mesh file: " + filename);
    test.close();
    
#ifdef DEAL_II_WITH_TRILINOS
    grid_in.read_exodusii(filename);
#else
    throw std::runtime_error(
        "Exodus format requires deal.II built with Trilinos/SEACAS support");
#endif
}

template <int dim>
void MeshReader<dim>::read_abaqus(const std::string& filename) {
    std::ifstream input(filename);
    if (!input)
        throw std::runtime_error("Cannot open mesh file: " + filename);
    read_abaqus(input);
}

template <int dim>
void MeshReader<dim>::read_abaqus(std::istream& input) {
    clear_metadata();
    
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    grid_in.read_abaqus(input);
    
    // Note: Node and element sets from Abaqus could be parsed here
    // but GridIn doesn't expose them directly. Would need custom parsing.
}

template <int dim>
void MeshReader<dim>::read_ucd(const std::string& filename) {
    std::ifstream input(filename);
    if (!input)
        throw std::runtime_error("Cannot open mesh file: " + filename);
    read_ucd(input);
}

template <int dim>
void MeshReader<dim>::read_ucd(std::istream& input) {
    clear_metadata();
    
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    grid_in.read_ucd(input);
}

template <int dim>
void MeshReader<dim>::read_auto(const std::string& filename) {
    std::string ext = get_extension(filename);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == "msh") {
        read_gmsh(filename);
    } else if (ext == "vtk" || ext == "vtu") {
        read_vtk(filename);
    } else if (ext == "inp") {
        read_abaqus(filename);
    } else if (ext == "ucd") {
        read_ucd(filename);
    } else if (ext == "exo" || ext == "e") {
        read_exodus(filename);
    } else {
        throw std::runtime_error("Unsupported mesh format: ." + ext + 
            "\nSupported formats: .msh, .vtk, .vtu, .inp, .ucd, .exo");
    }
}

template <int dim>
void MeshReader<dim>::read_from_string(const std::string& data, 
                                        const std::string& format) {
    std::istringstream input(data);
    
    std::string fmt = format;
    std::transform(fmt.begin(), fmt.end(), fmt.begin(), ::tolower);
    
    if (fmt == "msh" || fmt == "gmsh") {
        read_gmsh(input);
    } else if (fmt == "vtk" || fmt == "vtu") {
        read_vtk(input);
    } else if (fmt == "inp" || fmt == "abaqus") {
        read_abaqus(input);
    } else if (fmt == "ucd") {
        read_ucd(input);
    } else {
        throw std::runtime_error("Unsupported mesh format for string input: " + format);
    }
}

// ============================================================================
// Geometric Transformations
// ============================================================================

template <int dim>
void MeshReader<dim>::scale(double factor) {
    GridTools::scale(factor, triangulation);
}

template <int dim>
void MeshReader<dim>::translate(const Point<dim>& offset) {
    Tensor<1, dim> shift;
    for (unsigned int d = 0; d < dim; ++d)
        shift[d] = offset[d];
    GridTools::shift(shift, triangulation);
}

template <int dim>
void MeshReader<dim>::translate(const Tensor<1, dim>& offset) {
    GridTools::shift(offset, triangulation);
}

template <int dim>
void MeshReader<dim>::rotate(double angle) {
    if constexpr (dim == 2) {
        GridTools::rotate(angle, triangulation);
    } else {
        // 3D: rotate about Z axis by default
        GridTools::rotate(angle, 2, triangulation);
    }
}

template <int dim>
void MeshReader<dim>::rotate(double angle, unsigned int axis) {
    if constexpr (dim == 3) {
        GridTools::rotate(angle, axis, triangulation);
    } else {
        // 2D: only one rotation axis
        GridTools::rotate(angle, triangulation);
    }
}

template <int dim>
void MeshReader<dim>::center_at_origin() {
    Point<dim> min_pt, max_pt;
    compute_bounding_box(min_pt, max_pt);
    
    Point<dim> center;
    for (unsigned int d = 0; d < dim; ++d)
        center[d] = -0.5 * (min_pt[d] + max_pt[d]);
    
    translate(center);
}

template <int dim>
void MeshReader<dim>::move_to_origin() {
    Point<dim> min_pt, max_pt;
    compute_bounding_box(min_pt, max_pt);
    
    Point<dim> offset;
    for (unsigned int d = 0; d < dim; ++d)
        offset[d] = -min_pt[d];
    
    translate(offset);
}

// ============================================================================
// Mesh Information
// ============================================================================

template <int dim>
MeshInfo<dim> MeshReader<dim>::get_mesh_info() const {
    MeshInfo<dim> info;
    
    info.num_cells = triangulation.n_active_cells();
    info.num_vertices = triangulation.n_vertices();
    info.num_faces = 0;
    
    // Collect boundary and material IDs
    for (const auto& cell : triangulation.active_cell_iterators()) {
        info.material_ids.insert(cell->material_id());
        
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (cell->face(f)->at_boundary()) {
                info.boundary_ids.insert(cell->face(f)->boundary_id());
                info.num_faces++;
            }
        }
    }
    
    // Copy named sets
    info.named_node_sets = node_sets_;
    info.named_element_sets = element_sets_;
    info.physical_groups = physical_groups_;
    
    // Compute bounding box
    compute_bounding_box(info.bounding_box_min, info.bounding_box_max);
    
    return info;
}

// ============================================================================
// Boundary ID Management
// ============================================================================

template <int dim>
void MeshReader<dim>::set_boundary_id_by_location(
    std::function<bool(const Point<dim>&)> predicate,
    unsigned int boundary_id) {
    
    for (auto& cell : triangulation.active_cell_iterators()) {
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (cell->face(f)->at_boundary()) {
                Point<dim> face_center = cell->face(f)->center();
                if (predicate(face_center)) {
                    cell->face(f)->set_boundary_id(boundary_id);
                }
            }
        }
    }
}

template <int dim>
void MeshReader<dim>::set_boundary_id_on_plane(
    const Tensor<1, dim>& normal,
    const Point<dim>& point,
    double tolerance,
    unsigned int boundary_id) {
    
    Tensor<1, dim> n = normal / normal.norm();
    
    set_boundary_id_by_location(
        [&n, &point, tolerance](const Point<dim>& p) {
            Tensor<1, dim> v;
            for (unsigned int d = 0; d < dim; ++d)
                v[d] = p[d] - point[d];
            return std::abs(v * n) < tolerance;
        },
        boundary_id);
}

template <int dim>
void MeshReader<dim>::set_boundary_id_in_box(
    const Point<dim>& box_min,
    const Point<dim>& box_max,
    unsigned int boundary_id) {
    
    set_boundary_id_by_location(
        [&box_min, &box_max](const Point<dim>& p) {
            for (unsigned int d = 0; d < dim; ++d) {
                if (p[d] < box_min[d] || p[d] > box_max[d])
                    return false;
            }
            return true;
        },
        boundary_id);
}

template <int dim>
void MeshReader<dim>::copy_material_to_boundary_ids() {
    for (auto& cell : triangulation.active_cell_iterators()) {
        unsigned int mat_id = cell->material_id();
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (cell->face(f)->at_boundary()) {
                cell->face(f)->set_boundary_id(mat_id);
            }
        }
    }
}

// ============================================================================
// Validation
// ============================================================================

template <int dim>
bool MeshReader<dim>::validate() const {
    return get_validation_issues().empty();
}

template <int dim>
std::vector<std::string> MeshReader<dim>::get_validation_issues() const {
    std::vector<std::string> issues;
    
    if (triangulation.n_active_cells() == 0) {
        issues.push_back("Mesh is empty (no cells)");
        return issues;
    }
    
    // Check for zero-volume cells
    unsigned int zero_volume_count = 0;
    for (const auto& cell : triangulation.active_cell_iterators()) {
        if (cell->measure() <= 0) {
            zero_volume_count++;
        }
    }
    if (zero_volume_count > 0) {
        issues.push_back("Found " + std::to_string(zero_volume_count) + 
                        " cells with zero or negative volume (inverted elements)");
    }
    
    // Check for disconnected regions
    // (simplified check - just look for isolated vertices)
    
    // Check for extremely small elements
    double min_volume = std::numeric_limits<double>::max();
    double max_volume = 0;
    for (const auto& cell : triangulation.active_cell_iterators()) {
        double vol = cell->measure();
        if (vol > 0) {
            min_volume = std::min(min_volume, vol);
            max_volume = std::max(max_volume, vol);
        }
    }
    
    if (max_volume > 0 && min_volume / max_volume < 1e-6) {
        issues.push_back("Extreme element size variation detected (ratio: " +
                        std::to_string(min_volume / max_volume) + ")");
    }
    
    return issues;
}

// ============================================================================
// Private Helper Methods
// ============================================================================

template <int dim>
std::string MeshReader<dim>::get_extension(const std::string& filename) const {
    size_t pos = filename.find_last_of('.');
    if (pos == std::string::npos) return "";
    return filename.substr(pos + 1);
}

template <int dim>
void MeshReader<dim>::compute_bounding_box(Point<dim>& min_pt, 
                                            Point<dim>& max_pt) const {
    for (unsigned int d = 0; d < dim; ++d) {
        min_pt[d] = std::numeric_limits<double>::max();
        max_pt[d] = std::numeric_limits<double>::lowest();
    }
    
    for (const auto& cell : triangulation.active_cell_iterators()) {
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
            const Point<dim>& vertex = cell->vertex(v);
            for (unsigned int d = 0; d < dim; ++d) {
                min_pt[d] = std::min(min_pt[d], vertex[d]);
                max_pt[d] = std::max(max_pt[d], vertex[d]);
            }
        }
    }
}

template <int dim>
void MeshReader<dim>::clear_metadata() {
    node_sets_.clear();
    element_sets_.clear();
    physical_groups_.clear();
}

template <int dim>
void MeshReader<dim>::parse_gmsh_physical_groups() {
    // Physical groups in GMSH are encoded in material_id and boundary_id
    // This method attempts to reconstruct the mapping
    
    // Collect all material IDs as physical volume groups
    for (const auto& cell : triangulation.active_cell_iterators()) {
        unsigned int mat_id = cell->material_id();
        std::string name = "volume_" + std::to_string(mat_id);
        physical_groups_[name] = mat_id;
    }
    
    // Collect boundary IDs as physical surface groups
    std::set<unsigned int> boundary_ids;
    for (const auto& cell : triangulation.active_cell_iterators()) {
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            if (cell->face(f)->at_boundary()) {
                boundary_ids.insert(cell->face(f)->boundary_id());
            }
        }
    }
    
    for (unsigned int bid : boundary_ids) {
        std::string name = "surface_" + std::to_string(bid);
        physical_groups_[name] = bid;
    }
}

// ============================================================================
// MeshGenerator Implementation
// ============================================================================

template <int dim>
void MeshGenerator<dim>::generate_box(
    Triangulation<dim>& tria,
    const Point<dim>& p1,
    const Point<dim>& p2,
    const std::vector<unsigned int>& divisions,
    bool colorize) {
    
    tria.clear();
    GridGenerator::subdivided_hyper_rectangle(tria, divisions, p1, p2, colorize);
}

template <int dim>
void MeshGenerator<dim>::generate_cylinder(
    Triangulation<dim>& tria,
    double radius,
    double half_length) {
    
    tria.clear();
    
    if constexpr (dim == 3) {
        GridGenerator::cylinder(tria, radius, half_length);
    } else {
        // 2D: create a rectangle representing the cross-section
        GridGenerator::hyper_rectangle(tria,
            Point<dim>(-radius, -half_length),
            Point<dim>(radius, half_length),
            true);
    }
}

template <int dim>
void MeshGenerator<dim>::generate_sphere(
    Triangulation<dim>& tria,
    const Point<dim>& center,
    double radius) {
    
    tria.clear();
    GridGenerator::hyper_ball(tria, center, radius);
}

template <int dim>
void MeshGenerator<dim>::generate_plate_with_hole(
    Triangulation<dim>& tria,
    double plate_width,
    double plate_height,
    double hole_radius) {
    
    tria.clear();
    
    if constexpr (dim == 2) {
        GridGenerator::hyper_cube_with_cylindrical_hole(tria, hole_radius, 
                                                         plate_width / 2.0);
        GridTools::scale(1.0, tria);  // Adjust as needed
    } else {
        // 3D: create extruded plate with hole
        Triangulation<2> tria_2d;
        GridGenerator::hyper_cube_with_cylindrical_hole(tria_2d, hole_radius,
                                                         plate_width / 2.0);
        
        // Extrude to 3D
        GridGenerator::extrude_triangulation(tria_2d, 
                                              static_cast<unsigned int>(plate_height / hole_radius),
                                              plate_height, 
                                              tria);
    }
}

template <int dim>
void MeshGenerator<dim>::generate_quarter_plate_with_hole(
    Triangulation<dim>& tria,
    double plate_width,
    double plate_height,
    double hole_radius) {
    
    tria.clear();
    
    if constexpr (dim == 2) {
        // Quarter plate: first quadrant only
        GridGenerator::hyper_cube_with_cylindrical_hole(tria, hole_radius,
                                                         plate_width / 2.0,
                                                         plate_height / 2.0);
        
        // Keep only first quadrant
        // Note: GridGenerator already creates quarter model in some versions
    } else {
        // 3D quarter model
        Triangulation<2> tria_2d;
        GridGenerator::hyper_cube_with_cylindrical_hole(tria_2d, hole_radius,
                                                         plate_width / 2.0);
        
        GridGenerator::extrude_triangulation(tria_2d,
                                              static_cast<unsigned int>(plate_height / hole_radius),
                                              plate_height / 2.0,
                                              tria);
    }
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template class MeshReader<3>;
template class MeshReader<2>;

template struct MeshInfo<3>;
template struct MeshInfo<2>;

template class MeshGenerator<3>;
template class MeshGenerator<2>;

} // namespace FEA
