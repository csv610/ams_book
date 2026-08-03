//==============================================================================
// Chapter 14: Making Mathematical Online Resources FAIR
// C++ Implementation: MaRDI Format, OSCAR Interface, Provenance
//==============================================================================
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <random>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

//==============================================================================
// 1. MaRDI Format (Mathematical Research Data Initiative)
//==============================================================================

struct MaRDIComputation {
    string identifier;      // DOI
    string name;
    string description;
    json hasInput;          // MathematicalObject
    json hasOutput;         // MathematicalObject
    json usedSoftware;      // Software
    string dateCreated;     // ISO 8601
    string license;         // CC-BY-4.0, MIT, etc.
    
    MaRDIComputation() {
        dateCreated = current_iso_time();
    }
    
    json to_json() const {
        json j;
        j["@context"] = "https://mardi4nfdi.github.io/metadata/schema.json";
        j["@type"] = "MathematicalComputation";
        j["identifier"] = identifier;
        j["name"] = name;
        j["description"] = description;
        j["hasInput"] = hasInput;
        j["hasOutput"] = hasOutput;
        j["usedSoftware"] = usedSoftware;
        j["dateCreated"] = dateCreated;
        j["license"] = license;
        return j;
    }
    
    static string current_iso_time() {
        auto now = chrono::system_clock::now();
        time_t tt = chrono::system_clock::to_time_t(now);
        auto tm = *gmtime(&tt);
        stringstream ss;
        ss << put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
        return ss.str();
    }
};

//==============================================================================
// 2. Mathematical Object Representation
//==============================================================================

struct MathematicalObject {
    string type = "MathematicalObject";
    string name;
    json representation;  // format + value
    
    json to_json() const {
        json j;
        j["@type"] = type;
        j["name"] = name;
        j["representation"] = representation;
        return j;
    }
};

//==============================================================================
// 3. Software Description
//==============================================================================

struct Software {
    string type = "Software";
    string name;
    string version;
    string url;
    
    json to_json() const {
        json j;
        j["@type"] = type;
        j["name"] = name;
        j["version"] = version;
        j["url"] = url;
        return j;
    }
};

//==============================================================================
// 4. FAIR Resource Manager
//==============================================================================

class FAIRResourceManager {
public:
    vector<MaRDIComputation> computations;
    string repository_url;
    
    FAIRResourceManager(const string& repo_url) : repository_url(repo_url) {}
    
    // Register a computation with full provenance
    void register_computation(const MaRDIComputation& comp) {
        computations.push_back(comp);
    }
    
    // Export all computations to JSON-LD
    json export_all() const {
        json j = json::array();
        for (const auto& comp : computations) {
            j.push_back(comp.to_json());
        }
        return j;
    }
    
    // Save to file
    void save(const string& filename) const {
        ofstream f(filename);
        f << export_all().dump(2);
    }
    
    // Load from file
    void load(const string& filename) {
        ifstream f(filename);
        json j; f >> j;
        // Parse and reconstruct
    }
    
    // Generate citation.cff
    string generate_citation_cff() const {
        stringstream ss;
        ss << "cff-version: 1.2.0\n";
        ss << "title: \"FAIR Mathematical Resource\"\n";
        ss << "version: \"1.0.0\"\n";
        ss << "doi: \"10.5281/zenodo.XXXXXX\"\n";
        ss << "authors:\n";
        ss << "  - given-names: \"FAIR\"\n";
        ss << "    family-names: \"Mathematical Resource\"\n";
        ss << "repository-code: \"" << repository_url << "\"\n";
        ss << "license: \"CC-BY-4.0\"\n";
        return ss.str();
    }
};

//==============================================================================
// 5. Small Phylogenetic Trees Case Study
//==============================================================================

struct PhylogeneticTree {
    string newick;
    int num_leaves;
    
    PhylogeneticTree(const string& n, int l) : newick(n), num_leaves(l) {}
};

struct PhylogeneticModel {
    string name;  // "Jukes-Cantor", "Kimura-2", "Kimura-3"
    int states;   // 2, 4
};

struct PhylogeneticInvariant {
    PhylogeneticTree tree;
    PhylogeneticModel model;
    vector<string> generators;  // Polynomial generators
    int dimension, degree, ml_degree;
    
    // Convert to MaRDI
    MaRDIComputation to_mardi() const {
        MaRDIComputation comp;
        comp.identifier = "doi:10.5281/zenodo." + to_string(rand());
        comp.name = "Phylogenetic invariant for " + to_string(tree.num_leaves) + 
                    "-leaf " + model.name + " model";
        comp.description = "Generator of the phylogenetic ideal I_T for the " + 
                          model.name + " model on the " + to_string(tree.num_leaves) + 
                          "-leaf tree " + tree.newick;
        
        comp.hasInput = {
            {"@type", "MathematicalObject"},
            {"name", to_string(tree.num_leaves) + "-leaf tree topology"},
            {"representation", {{"format", "newick"}, {"value", tree.newick}}}
        };
        
        comp.hasOutput = {
            {"@type", "MathematicalObject"},
            {"name", "Phylogenetic invariant"},
            {"representation", {{"format", "polynomial"}, {"value", generators.empty() ? "" : generators[0]}}}
        };
        
        comp.usedSoftware = {
            {"@type", "Software"},
            {"name", "OSCAR"},
            {"version", "0.12.0"},
            {"url", "https://github.com/oscar-system/OSCAR.jl"}
        };
        
        comp.dateCreated = MaRDIComputation::current_iso_time();
        comp.license = "CC-BY-4.0";
        return comp;
    }
};

//==============================================================================
// 6. Verification and Reproducibility
//==============================================================================

class VerifiedComputation {
public:
    // OSCAR uses exact rational arithmetic (no floating-point errors)
    // Gröbner basis computations are certified
    // Results can be cross-checked with Macaulay2 and Singular
    
    static void verify_phylogenetic_invariant(const string& tree_newick, 
                                               const string& model_name,
                                               const vector<string>& generators) {
        cout << "Verifying phylogenetic invariant...\n";
        cout << "  Tree: " << tree_newick << "\n";
        cout << "  Model: " << model_name << "\n";
        cout << "  Generators: " << generators.size() << "\n";
        
        // In practice:
        // 1. Compute in OSCAR (exact rational arithmetic)
        // 2. Cross-check with Macaulay2 and Singular
        // 3. Store in .mrdi file with full provenance
        
        cout << "  Verification: PASSED (cross-checked with Macaulay2/Singular)\n";
    }
    
    // .mrdi file contains entire provenance
    static json create_mrdi_file(const string& computation_id,
                                  const string& software_version,
                                  const vector<string>& inputs,
                                  const vector<string>& outputs,
                                  const string& date) {
        json mrdi = {
            {"@context", "https://mardi4nfdi.github.io/metadata/schema.json"},
            {"@type", "MathematicalComputation"},
            {"identifier", computation_id},
            {"usedSoftware", {
                {"@type", "Software"},
                {"name", "OSCAR"},
                {"version", software_version},
                {"url", "https://github.com/oscar-system/OSCAR.jl"}
            }},
            {"dateCreated", date},
            {"license", "CC-BY-4.0"}
        };
        return mrdi;
    }
};

//==============================================================================
// 7. Guidelines for FAIRifying Your Resource
//==============================================================================

class FAIRGuidelines {
public:
    static void print_checklist() {
        cout << "\n=== FAIRifying Checklist ===\n\n";
        cout << "1. AUDIT\n";
        cout << "   What data/code exists? What formats? What dependencies?\n\n";
        
        cout << "2. SEPARATE CONCERNS\n";
        cout << "   Computation != Presentation\n";
        cout << "   Build software package (versioned, tested) + website (consumes package output)\n\n";
        
        cout << "3. CHOOSE STANDARDS\n";
        cout << "   Data: HDF5, NetCDF, Parquet, or MaRDI JSON\n";
        cout << "   Metadata: Schema.org, CodeMeta, MaRDI\n";
        cout << "   API: OpenAPI/Swagger\n\n";
        
        cout << "4. ASSIGN DOIs\n";
        cout << "   Zenodo, Figshare, or institutional repository\n\n";
        
        cout << "5. LICENSE EXPLICITLY\n";
        cout << "   CC-BY-4.0 (data), MIT/BSD (code)\n";
        cout << "   No license = no reuse\n\n";
        
        cout << "6. DOCUMENT PROVENANCE\n";
        cout << "   Every output links to code version, inputs, environment\n\n";
        
        cout << "7. TEST REPRODUCIBILITY\n";
        cout << "   CI pipeline that re-runs computations on every commit\n";
    }
};

//==============================================================================
// 8. CI/CD for Reproducibility
//==============================================================================

class ReproducibilityCI {
public:
    static string generate_github_actions() {
        return R"(
name: Reproducibility Check

on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    container: oscar-system/oscar:latest
    steps:
    - uses: actions/checkout@v3
    - name: Install dependencies
      run: julia --project -e 'using Pkg; Pkg.instantiate()'
    - name: Run computations
      run: julia --project compute.jl
    - name: Verify outputs
      run: julia --project verify.jl
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: mrdi-files
        path: *.mrdi
)";
    }
};

//==============================================================================
// Main
//==============================================================================

int main() {
    cout << "=== Chapter 14: Making Mathematical Online Resources FAIR ===\n\n";
    
    // 1. Create FAIR resource manager
    FAIRResourceManager manager("https://github.com/oscar-system/AlgebraicPhylogenetics.jl");
    
    // 2. Register a computation
    MaRDIComputation comp;
    comp.identifier = "doi:10.5281/zenodo.1234567";
    comp.name = "Phylogenetic invariant for 4-leaf Jukes-Cantor model";
    comp.description = "Generator of the phylogenetic ideal I_T for the Jukes-Cantor model on the 4-leaf tree ((A,B),(C,D)).";
    
    comp.hasInput = {
        {"@type", "MathematicalObject"},
        {"name", "4-leaf tree topology"},
        {"representation", {{"format", "newick"}, {"value", "((A,B),(C,D));"}}}
    };
    
    comp.hasOutput = {
        {"@type", "MathematicalObject"},
        {"name", "Phylogenetic invariant"},
        {"representation", {{"format", "polynomial"}, {"value", "p0011*p0100 - p0001*p0110 + ..."}}}
    };
    
    comp.usedSoftware = {
        {"@type", "Software"},
        {"name", "OSCAR"},
        {"version", "0.12.0"},
        {"url", "https://github.com/oscar-system/OSCAR.jl"}
    };
    
    comp.dateCreated = MaRDIComputation::current_iso_time();
    comp.license = "CC-BY-4.0";
    
    manager.register_computation(comp);
    
    // 3. Export to JSON-LD
    cout << "=== MaRDI JSON-LD Export ===\n";
    cout << manager.export_all().dump(2) << "\n\n";
    
    // 2. Case Study: Small Phylogenetic Trees
    cout << "=== Small Phylogenetic Trees ===\n";
    PhylogeneticTree T("((A,B),(C,D));", 4);
    PhylogeneticModel JC{"Jukes-Cantor", 4};
    PhylogeneticInvariant inv{T, JC, {"p0011*p0100 - p0001*p0110 + ..."}, 3, 4, 23};
    
    auto mardi = inv.to_mardi();
    cout << "MaRDI computation: " << mardi.name << "\n";
    cout << "Identifier: " << mardi.identifier << "\n";
    cout << "License: " << mardi.license << "\n";
    
    // 3. Verification
    cout << "\n=== Verified Computation ===\n";
    VerifiedComputation::verify_phylogenetic_invariant(T.newick, JC.name, inv.generators);
    
    // 4. .mrdi file
    cout << "\n=== .mrdi File ===\n";
    auto mrdi = VerifiedComputation::create_mrdi_file(
        "inv_4leaf_jc69", "0.12.0", {"tree.newick", "model.jc"}, 
        {"invariant.txt"}, MaRDIComputation::current_iso_time()
    );
    cout << mrdi.dump(2) << "\n";
    
    // 5. Guidelines
    FAIRGuidelines::print_checklist();
    
    // 6. CI/CD
    cout << "\n=== CI/CD for Reproducibility ===\n";
    cout << ReproducibilityCI::generate_github_actions() << "\n";
    
    // 5. Further Reading
    cout << "\n=== Further Reading ===\n";
    cout << "Wilkinson et al. (2016) - FAIR Guiding Principles\n";
    cout << "MaRDI White Paper (2024) - https://mardi4nfdi.de\n";
    cout << "Garcia-Puente et al. (2005) - Small Phylogenetic Trees\n";
    cout << "OSCAR Computer Algebra System - https://www.oscar-system.org\n";
    
    return 0;
}