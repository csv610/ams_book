//==============================================================================
// Chapter 12: Operads for Designing Systems of Systems
// C++ Implementation: Operad Structure, Network Operads, Algebras
//==============================================================================
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <memory>
#include <functional>

using namespace std;

//==============================================================================
// 1. Basic Operad Structure
//==============================================================================

// Type (color) in an operad
using Type = int;  // Simplified: types are natural numbers

// Operation: input types -> output type
struct Operation {
    vector<Type> inputs;
    Type output;
    string name;  // For display
    
    Operation(const vector<Type>& in, Type out, const string& n = "")
        : inputs(in), output(out), name(n) {}
    
    int arity() const { return inputs.size(); }
};

// Operad: collection of operations with composition
class Operad {
public:
    map<pair<vector<Type>, Type>, vector<Operation>> operations;
    
    void add(const Operation& op) {
        operations[{op.inputs, op.output}].push_back(op);
    }
    
    // Get operations with given input/output types
    vector<Operation> get(const vector<Type>& inputs, Type output) const {
        auto it = operations.find({inputs, output});
        return (it != operations.end()) ? it->second : vector<Operation>{};
    }
    
    // Composition: f ∘ (g_1, ..., g_n)
    // f: X1,...,Xn -> Y
    // g_i: Z_i1,...,Z_iki -> X_i
    // Result: Z_11,...,Z_nk_n -> Y
    Operation compose(const Operation& f, const vector<Operation>& gs) const {
        if (f.inputs.size() != gs.size()) {
            throw runtime_error("Arity mismatch in composition");
        }
        
        vector<Type> new_inputs;
        for (const auto& g : gs) {
            new_inputs.insert(new_inputs.end(), g.inputs.begin(), g.inputs.end());
        }
        
        // Verify types match
        for (size_t i = 0; i < gs.size(); ++i) {
            if (gs[i].output != f.inputs[i]) {
                throw runtime_error("Type mismatch in composition");
            }
        }
        
        return Operation(new_inputs, f.output, 
                        f.name + " ∘ (" + join_names(gs) + ")");
    }
    
    // Identity operation
    Operation identity(Type X) const {
        return Operation({X}, X, "id_" + to_string(X));
    }
    
private:
    string join_names(const vector<Operation>& ops) const {
        string s;
        for (size_t i = 0; i < ops.size(); ++i) {
            if (i > 0) s += ", ";
            s += ops[i].name;
        }
        return s;
    }
};

//==============================================================================
// 2. Algebra of an Operad
//==============================================================================

// Algebra: assigns to each type X a set A(X), and to each operation
// a function A(f): A(X1) × ... × A(Xn) -> A(Y)
template<typename Value>
class Algebra {
public:
    map<Type, vector<Value>> type_values;
    map<string, function<Value(const vector<Value>&)>> op_functions;
    
    // Assign values to type
    void set_type(Type X, const vector<Value>& vals) {
        type_values[X] = vals;
    }
    
    // Define operation semantics
    void define_op(const string& op_name, 
                   function<Value(const vector<Value>&)> f) {
        op_functions[op_name] = f;
    }
    
    // Apply operation
    Value apply(const string& op_name, const vector<Value>& args) const {
        auto it = op_functions.find(op_name);
        if (it == op_functions.end()) {
            throw runtime_error("Operation not defined: " + op_name);
        }
        return it->second(args);
    }
};

//==============================================================================
// 3. Network Operad (Aircraft Communication)
//==============================================================================

// Network operad: types = number of aircraft
// Operations = graphs with input/output vertices
struct NetworkOperad : public Operad {
    // Add operation: connect networks by adding edges
    void add_network_op(const vector<int>& in_sizes, int out_size, 
                        const vector<pair<int,int>>& edges, 
                        const string& name = "") {
        vector<Type> inputs;
        for (int s : in_sizes) inputs.push_back(s);
        Operation op(inputs, out_size, name);
        // Store edges separately (simplified)
        add(op);
    }
};

// Algebra: actual aircraft states with positions and comms ranges
struct AircraftState {
    Vector3d position;
    Vector3d velocity;
    double comm_range = 1000.0;  // meters
    
    bool in_range(const AircraftState& other) const {
        return (position - other.position).norm() <= comm_range;
    }
};

class NetworkAlgebra {
public:
    // Interpret type n as set of n aircraft states
    vector<vector<AircraftState>> type_states;
    
    vector<AircraftState> interpret_type(int n) const {
        if (n >= type_states.size()) return {};
        return type_states[n];
    }
    
    // Apply network operation: add comms links between in-range aircraft
    vector<AircraftState> apply_connect(const vector<AircraftState>& states,
                                        const vector<pair<int,int>>& new_edges) {
        vector<AircraftState> result = states;
        for (auto [i, j] : new_edges) {
            if (i < result.size() && j < result.size()) {
                // In real implementation, add comms link
            }
        }
        return result;
    }
};

//==============================================================================
// 3. Search and Rescue Operad
//==============================================================================

struct Asset {
    string type;       // "ship", "helicopter", "boat", "drone"
    int capacity = 0;  // Number of sub-assets it can carry
    Vector3d position;
    double fuel = 1000.0;
};

struct SAROperad : public Operad {
    // Types = number of assets at each level
    // Operations = ways to nest assets
    
    void add_nesting_op(int carrier_capacity, int num_carried, const string& name) {
        // Operation: carrier + carried -> combined
        // Input: 1 (carrier) + num_carried (carried) -> Output: 1 (combined unit)
        vector<Type> inputs = {1};  // Carrier
        for (int i = 0; i < num_carried; ++i) inputs.push_back(1);
        add(Operation(inputs, 1, name));
    }
};

//==============================================================================
// 4. Catalyst Operads (for task planning)
//==============================================================================

struct CatalystOperad : public Operad {
    // Some operations require a "catalyst" agent to enable composition
    // e.g., a coordinator helicopter enables boat deployment
    
    void add_catalyzed_op(const string& catalyst, 
                          const vector<Type>& inputs, Type output,
                          const string& name) {
        // In full implementation, track catalyst requirement
        add(Operation(inputs, output, name + "_catalyzed_by_" + catalyst));
    }
};

//==============================================================================
// 5. Levels of Abstraction
//==============================================================================

enum AbstractionLevel { ABSTRACT, CONCRETE, DETAILED };

// Operad homomorphism: map between operads preserving structure
struct OperadHomomorphism {
    map<Type, Type> type_map;
    map<string, string> op_map;
    
    // Map operation from source to target operad
    Operation map_op(const Operation& op, const Operad& source, 
                     Operad& target) const {
        vector<Type> mapped_inputs;
        for (Type t : op.inputs) {
            auto it = type_map.find(t);
            if (it == type_map.end()) throw runtime_error("Type not mapped");
            mapped_inputs.push_back(it->second);
        }
        
        Type mapped_output = type_map.at(op.output);
        string mapped_name = op_map.count(op.name) ? op_map.at(op.name) : op.name;
        
        return Operation(mapped_inputs, mapped_output, mapped_name);
    }
};

//==============================================================================
// 6. Constraint Programming Translation
//==============================================================================

struct ConstraintProgram {
    // Each operation -> decision variables
    // Composition -> constraints
    
    struct Variable {
        string name;
        enum Domain { BINARY, INTEGER, REAL } domain;
        double lb = 0, ub = 1;
    };
    
    vector<Variable> variables;
    vector<string> constraints;
    
    void add_operation_vars(const Operation& op) {
        // For each possible edge in operation, add binary variable
        // In network operad: variable for each possible connection
    }
    
    void add_composition_constraints(const Operation& outer,
                                     const vector<Operation>& inners) {
        // Composition = logical AND of inner constraints
        // Plus outer constraints on combined structure
    }
};

//==============================================================================
// Main
//==============================================================================

int main() {
    cout << "=== Chapter 12: Operads for Designing Systems of Systems ===\n\n";
    
    // 1. Basic Operad
    cout << "=== Basic Operad ===\n";
    Operad O;
    O.add(Operation({1, 2}, 3, "f"));
    O.add(Operation({3}, 4, "g"));
    O.add(Operation({1, 2}, 4, "g ∘ f"));
    
    auto f = O.get({1, 2}, 3)[0];
    auto g = O.get({3}, 4)[0];
    auto gf = O.compose(f, {g});
    cout << "Composed: " << gf.name << " : {1,2} -> 4\n";
    
    // Identity
    auto id = O.identity(1);
    cout << "Identity: " << id.name << " : {1} -> 1\n";
    
    // 2. Network Operad (Aircraft Comms)
    cout << "\n=== Network Operad ===\n";
    NetworkOperad net_op;
    net_op.add_network_op({2, 3}, 5, {{0,2}, {1,3}, {1,4}}, "connect_2_and_3");
    
    NetworkAlgebra alg;
    // Add some aircraft states
    AircraftState a1, a2, a3, a4, a5;
    a1.position = Vector3d(0, 0, 1000);
    a2.position = Vector3d(100, 0, 1000);
    a3.position = Vector3d(500, 0, 1000);
    a4.position = Vector3d(0, 500, 1000);
    a5.position = Vector3d(100, 500, 1000);
    a1.comm_range = a2.comm_range = a3.comm_range = a4.comm_range = a5.comm_range = 1000;
    
    // Check ranges
    cout << "Aircraft 1 in range of 2: " << a1.in_range(a2) << "\n";
    cout << "Aircraft 1 in range of 3: " << a1.in_range(a3) << "\n";
    
    // 3. Search and Rescue
    cout << "\n=== Search and Rescue Operad ===\n";
    SAROperad sar;
    sar.add_nesting_op(3, 3, "helicopter_carries_3_boats");
    sar.add_nesting_op(1, 2, "ship_carries_2_helos");
    
    cout << "Operations:\n";
    for (auto& [key, ops] : sar.operations) {
        for (auto& op : ops) {
            cout << "  " << op.name << " : ";
            for (int t : op.inputs) cout << t << " ";
            cout << "-> " << op.output << "\n";
        }
    }
    
    // 4. Search and Rescue Scenario
    cout << "\n=== SAR Scenario ===\n";
    cout << "1 ship carries 2 helicopters, each carries 3 boats\n";
    cout << "Total: 1 ship + 2 helos + 6 boats = 9 assets\n";
    cout << "Operad operations model all valid nestings\n";
    cout << "Optimization: maximize search effort within budget\n";
    
    // 5. Levels of Abstraction
    cout << "\n=== Levels of Abstraction ===\n";
    cout << "1. Abstract: 'Network of boats with comms'\n";
    cout << "2. Concrete: '3 RHIBs, 2 helicopters, SATCOM'\n";
    cout << "3. Detailed: Specific models, frequencies, protocols\n";
    cout << "Operad homomorphisms formalize moving between levels\n";
    
    // 6. Constraint Programming
    cout << "\n=== Constraint Programming ===\n";
    cout << "Each operation -> decision variables\n";
    cout << "Composition -> constraints\n";
    cout << "Solve with MILP/CPLEX for optimal designs\n";
    
    // 7. Applications
    cout << "\n=== Applications ===\n";
    cout << "1. Maritime search and rescue (Fastnet 1979, Sydney-Hobart 1998)\n";
    cout << "2. Network design for aircraft comms\n";
    cout << "3. Multi-agent task planning with catalyst operads\n";
    cout << "4. System-of-systems engineering (DARPA CASCADE)\n";
    cout << "5. Wiring diagrams for digital circuits (Spivak)\n";
    
    return 0;
}