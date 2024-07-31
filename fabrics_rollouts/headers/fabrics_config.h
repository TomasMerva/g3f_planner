#pragma once

#include <iostream>
#include <sstream> 
#include <string>
#include <vector>


struct FabricsConfig
{
    const uint dim_states{2}; //q,qdot
    uint num_goals{0};
    uint num_goals_weights{0};
    uint num_obstacles{0};
    uint num_collision_link{0};
    uint num_planes{0};
    uint num_dofs{0};
    uint fabrics_input_size{0};
    const uint fabrics_output_size{1}; //qdot_cmd

    friend std::ostream& operator <<(std::ostream& os, FabricsConfig const& config)
    {
        return os << "Fabrics configuration\n---\n"
                  << "Number of state vectors: " << config.dim_states << '\n'
                  << "Number of goals: " << config.num_goals << '\n'
                  << "Number of goals' weights: " << config.num_goals_weights << '\n'
                  << "Number of obstacles: " << config.num_obstacles << '\n'
                  << "Number of collision links: " << config.num_collision_link << '\n'
                  << "Number of plane constraints: " << config.num_planes << '\n'
                  << "Number of degrees of freedom: " << config.num_dofs << '\n'
                  << "Number of input vectors to casadi funct: " << config.fabrics_input_size << '\n'
                  << "Number of output vectors from casadi funct: " << config.fabrics_output_size << '\n'
                  << "---\n";
    }
};


struct FabricsArgumentSize
{
    const uint goal_dim{3};
    const uint goal_weight_dim{1};
    const uint obstacle_dim{1};
    const uint obstacle_pos_dim{3};
    const uint coll_link_dim{1};
    const uint plane_constraint_dim{4};

};
