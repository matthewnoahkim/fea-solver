#ifndef MESH_READER_H
#define MESH_READER_H

/**
 * @file mesh_reader.h
 * @brief Mesh file format reader with extended metadata support
 * 
 * Provides mesh import functionality supporting multiple file formats:
 * - GMSH (.msh) - native GMSH format with physical groups
 * - VTK (.vtk, .vtu) - Visualization Toolkit formats
 * - Abaqus (.inp) - Abaqus input files
 * - UCD (.ucd) - AVS UCD format
 * - Exodus II (.exo, .e) - Sandia ExodusII format (requires library)
 * 
 * Also provides utilities for:
 * - Geometric transformations (scale, translate, rotate)
 * - Named node/element set extraction
 * - Mesh information queries
 */

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <nlohmann/json.hpp>
#include <string>
#include <map>
#include <vector>
#include <set>
#include <functional>

namespace FEA {

using namespace dealii;
using json = nlohmann::json;

// ============================================================================
// Mesh Information
// ============================================================================

/**
 * @brief Summary information about a loaded mesh
 */
template <int dim>
struct MeshInfo {
    unsigned int num_cells;             ///< Number of active cells
    unsigned int num_vertices;          ///< Number of unique vertices
    unsigned int num_faces;             ///< Number of boundary faces
    
    std::set<unsigned int> boundary_ids;    ///< All boundary IDs present
    std::set<unsigned int> material_ids;    ///< All material/region IDs present
    
    std::map<std::string, std::set<unsigned int>> named_node_sets;
    std::map<std::string, std::set<unsigned int>> named_element_sets;
    std::map<std::string, unsigned int> physical_groups;  ///< GMSH physical groups
    
    Point<dim> bounding_box_min;        ///< Minimum corner of bounding box
    Point<dim> bounding_box_max;        ///< Maximum corner of bounding box
    
    /**
     * @brief Get mesh dimensions (extent in each direction)
     */
    Tensor<1, dim> get_dimensions() const {
        Tensor<1, dim> dims;
        for (unsigned int d = 0; d < dim; ++d)
            dims[d] = bounding_box_max[d] - bounding_box_min[d];
        return dims;
    }
    
    /**
     * @brief Get mesh centroid
     */
    Point<dim> get_centroid() const {
        Point<dim> center;
        for (unsigned int d = 0; d < dim; ++d)
            center[d] = 0.5 * (bounding_box_min[d] + bounding_box_max[d]);
        return center;
    }
    
    /**
     * @brief Get characteristic length (diagonal of bounding box)
     */
    double get_characteristic_length() const {
        return bounding_box_min.distance(bounding_box_max);
    }
    
    json to_json() const {
        json j;
        j["num_cells"] = num_cells;
        j["num_vertices"] = num_vertices;
        j["num_faces"] = num_faces;
        j["boundary_ids"] = std::vector<unsigned int>(boundary_ids.begin(), boundary_ids.end());
        j["material_ids"] = std::vector<unsigned int>(material_ids.begin(), material_ids.end());
        
        std::vector<double> bb_min(dim), bb_max(dim);
        for (unsigned int d = 0; d < dim; ++d) {
            bb_min[d] = bounding_box_min[d];
            bb_max[d] = bounding_box_max[d];
        }
        j["bounding_box"] = {
            {"min", bb_min},
            {"max", bb_max}
        };
        
        j["characteristic_length"] = get_characteristic_length();
        
        return j;
    }
};

// ============================================================================
// Mesh Reader
// ============================================================================

/**
 * @brief Reads mesh files in various formats and provides mesh utilities
 * 
 * @tparam dim Spatial dimension (2 or 3)
 * 
 * Example usage:
 * @code
 * Triangulation<3> triangulation;
 * MeshReader<3> reader(triangulation);
 * 
 * reader.read_auto("model.msh");  // Auto-detect format
 * reader.scale(1e-3);             // Convert mm to m
 * 
 * auto info = reader.get_mesh_info();
 * std::cout << "Loaded " << info.num_cells << " elements\n";
 * @endcode
 */
template <int dim>
class MeshReader {
public:
    /**
     * @brief Construct reader attached to a triangulation
     */
    explicit MeshReader(Triangulation<dim>& tria);
    
    // =========================================================================
    // File Format Readers
    // =========================================================================
    
    /**
     * @brief Read GMSH format mesh
     * @param filename Path to .msh file
     */
    void read_gmsh(const std::string& filename);
    
    /**
     * @brief Read GMSH format from stream
     */
    void read_gmsh(std::istream& input);
    
    /**
     * @brief Read VTK format mesh
     * @param filename Path to .vtk or .vtu file
     */
    void read_vtk(const std::string& filename);
    
    /**
     * @brief Read VTK format from stream
     */
    void read_vtk(std::istream& input);
    
    /**
     * @brief Read Exodus II format mesh
     * @param filename Path to .exo or .e file
     * @note Requires deal.II built with Trilinos/SEACAS
     */
    void read_exodus(const std::string& filename);
    
    /**
     * @brief Read Abaqus input format mesh
     * @param filename Path to .inp file
     */
    void read_abaqus(const std::string& filename);
    
    /**
     * @brief Read Abaqus format from stream
     */
    void read_abaqus(std::istream& input);
    
    /**
     * @brief Read UCD format mesh
     * @param filename Path to .ucd file
     */
    void read_ucd(const std::string& filename);
    
    /**
     * @brief Read UCD format from stream
     */
    void read_ucd(std::istream& input);
    
    /**
     * @brief Auto-detect format from file extension and read
     * @param filename Path to mesh file
     */
    void read_auto(const std::string& filename);
    
    /**
     * @brief Read mesh from string with specified format
     * @param data Mesh file contents as string
     * @param format Format identifier ("msh", "vtk", "inp", "ucd")
     */
    void read_from_string(const std::string& data, const std::string& format);
    
    // =========================================================================
    // Geometric Transformations
    // =========================================================================
    
    /**
     * @brief Scale mesh uniformly
     * @param factor Scale factor (< 1 shrinks, > 1 enlarges)
     */
    void scale(double factor);
    
    /**
     * @brief Translate mesh
     * @param offset Translation vector
     */
    void translate(const Point<dim>& offset);
    
    /**
     * @brief Translate mesh by vector
     */
    void translate(const Tensor<1, dim>& offset);
    
    /**
     * @brief Rotate mesh about origin (2D: angle in radians)
     */
    void rotate(double angle);  // 2D rotation
    
    /**
     * @brief Rotate mesh about axis (3D)
     * @param angle Rotation angle in radians
     * @param axis Rotation axis (0=X, 1=Y, 2=Z)
     */
    void rotate(double angle, unsigned int axis);
    
    /**
     * @brief Move mesh so bounding box is centered at origin
     */
    void center_at_origin();
    
    /**
     * @brief Move mesh so bounding box minimum is at origin
     */
    void move_to_origin();
    
    // =========================================================================
    // Mesh Information
    // =========================================================================
    
    /**
     * @brief Get comprehensive mesh information
     */
    MeshInfo<dim> get_mesh_info() const;
    
    /**
     * @brief Get named node sets (from Abaqus/Exodus)
     */
    const std::map<std::string, std::set<unsigned int>>& get_node_sets() const {
        return node_sets_;
    }
    
    /**
     * @brief Get named element sets (from Abaqus/Exodus)
     */
    const std::map<std::string, std::set<unsigned int>>& get_element_sets() const {
        return element_sets_;
    }
    
    /**
     * @brief Get GMSH physical groups
     */
    const std::map<std::string, unsigned int>& get_physical_groups() const {
        return physical_groups_;
    }
    
    // =========================================================================
    // Boundary ID Management
    // =========================================================================
    
    /**
     * @brief Assign boundary ID to faces matching a geometric condition
     * @param predicate Function returning true for faces to mark
     * @param boundary_id ID to assign
     */
    void set_boundary_id_by_location(
        std::function<bool(const Point<dim>&)> predicate,
        unsigned int boundary_id);
    
    /**
     * @brief Assign boundary ID to faces on a plane
     * @param normal Plane normal direction
     * @param point Point on the plane
     * @param tolerance Distance tolerance
     * @param boundary_id ID to assign
     */
    void set_boundary_id_on_plane(
        const Tensor<1, dim>& normal,
        const Point<dim>& point,
        double tolerance,
        unsigned int boundary_id);
    
    /**
     * @brief Assign boundary ID to faces in a bounding box
     */
    void set_boundary_id_in_box(
        const Point<dim>& box_min,
        const Point<dim>& box_max,
        unsigned int boundary_id);
    
    /**
     * @brief Copy boundary IDs from material IDs (useful for some mesh formats)
     */
    void copy_material_to_boundary_ids();
    
    // =========================================================================
    // Validation
    // =========================================================================
    
    /**
     * @brief Check if mesh is valid (no inverted elements, proper connectivity)
     */
    bool validate() const;
    
    /**
     * @brief Get list of validation issues
     */
    std::vector<std::string> get_validation_issues() const;
    
private:
    Triangulation<dim>& triangulation;
    
    std::map<std::string, std::set<unsigned int>> node_sets_;
    std::map<std::string, std::set<unsigned int>> element_sets_;
    std::map<std::string, unsigned int> physical_groups_;
    
    /**
     * @brief Extract file extension from filename
     */
    std::string get_extension(const std::string& filename) const;
    
    /**
     * @brief Compute mesh bounding box
     */
    void compute_bounding_box(Point<dim>& min_pt, Point<dim>& max_pt) const;
    
    /**
     * @brief Clear stored metadata
     */
    void clear_metadata();
    
    /**
     * @brief Parse GMSH physical groups from mesh
     */
    void parse_gmsh_physical_groups();
};

// ============================================================================
// Mesh Generator Utilities
// ============================================================================

/**
 * @brief Simple mesh generation utilities
 */
template <int dim>
class MeshGenerator {
public:
    /**
     * @brief Generate box mesh with specified divisions
     */
    static void generate_box(
        Triangulation<dim>& tria,
        const Point<dim>& p1,
        const Point<dim>& p2,
        const std::vector<unsigned int>& divisions,
        bool colorize = true);
    
    /**
     * @brief Generate cylinder mesh (3D only)
     */
    static void generate_cylinder(
        Triangulation<dim>& tria,
        double radius,
        double half_length);
    
    /**
     * @brief Generate sphere mesh
     */
    static void generate_sphere(
        Triangulation<dim>& tria,
        const Point<dim>& center,
        double radius);
    
    /**
     * @brief Generate plate with hole (common benchmark)
     */
    static void generate_plate_with_hole(
        Triangulation<dim>& tria,
        double plate_width,
        double plate_height,
        double hole_radius);
    
    /**
     * @brief Generate quarter symmetry model of plate with hole
     */
    static void generate_quarter_plate_with_hole(
        Triangulation<dim>& tria,
        double plate_width,
        double plate_height,
        double hole_radius);
};

} // namespace FEA

#endif // MESH_READER_H
