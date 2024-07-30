#include "fabrics_server.h"
#include "pure_controller.c"


FabricsServer::FabricsServer(const uint port, const std::string config_file_path)
    : fabrics_config(FabricsServer::ReadConfigFile(config_file_path))
{
    FabricsServer::PrintFabricsArguments();
    FabricsServer::_ComputeParameterIndices();
    FabricsServer::PrintArgumentsIndices();

    _fabrics_input = new const double*[fabrics_config.fabrics_input_size];
    _fabrics_output  = new double*[fabrics_config.fabrics_output_size];

    _server = std::make_unique<TCPServer>(port);
}



void
FabricsServer::PrintFabricsArguments()
{
    std::cout << fabrics_config;
}

void
FabricsServer::PrintArgumentsIndices()
{
    std::cout << "Fabrics args indices\n---\n";
    for (const auto idx_pair : _fabrics_args_idx)
    {
        std::cout << idx_pair.first << ": " 
                  << idx_pair.second.first << " - "
                  << idx_pair.second.second <<"\n";
    }
}

FabricsConfig
FabricsServer::ReadConfigFile(const std::string config_file_path)
{
    YAML::Node config_file = YAML::LoadFile(config_file_path);
    FabricsConfig config;
    config.num_goals = config_file["problem"]["goal"]["goal_definition"].size();
    config.num_goals_weights = config_file["problem"]["goal"]["goal_definition"].size();
    config.num_obstacles = 
            config_file["problem"]["environment"]["number_spheres"]["dynamic"].as<double>() 
          + config_file["problem"]["environment"]["number_spheres"]["static"].as<double>();
    config.num_collision_link = config_file["problem"]["robot_representation"]["collision_links"].size();
    config.num_planes = config_file["problem"]["environment"]["number_planes"].size();
    config.num_dofs = config_file["problem"]["joint_limits"]["lower_limits"].size();

    //obstacle = radius + position
    config.fabrics_input_size = 
              fabrics_config.num_goals
            + fabrics_config.num_goals_weights
            + fabrics_config.num_obstacles*2 
            + fabrics_config.num_collision_link
            + fabrics_config.num_planes
            + fabrics_config.dim_states;
    return config;
}



void
FabricsServer::_ComputeParameterIndices()
{
    _fabrics_args_idx.clear();
    uint start_idx{0};

    // Plane constraint
    for (uint i=0; i<fabrics_config.num_planes; ++i)
    {
        uint end_idx{start_idx + _fabrics_args_size.plane_constraint_dim};

        std::string name = "plane_" + std::to_string(i);
        _fabrics_args_idx.push_back(constraint_idx_pair(name, std::pair<uint, uint>(start_idx, end_idx)));
        start_idx = end_idx;
    }

    // q and qdot
    for (uint i=0; i<fabrics_config.dim_states; ++i)
    {
        uint end_idx{start_idx + fabrics_config.num_dofs};
        std::string name = "q_state_" + std::to_string(i);
        _fabrics_args_idx.push_back(constraint_idx_pair(name, std::pair<uint, uint>(start_idx, end_idx)));        start_idx = end_idx;
    }
    // radius body
    for (uint i=0; i<fabrics_config.num_collision_link; ++i)
    {
        uint end_idx{start_idx + _fabrics_args_size.coll_link_dim};
        std::string name = "radius_body_" + std::to_string(i);
        _fabrics_args_idx.push_back(constraint_idx_pair(name, std::pair<uint, uint>(start_idx, end_idx)));
        start_idx = end_idx;
    }
    // radius obstacle
    for (uint i=0; i<fabrics_config.num_obstacles; ++i)
    {
        uint end_idx{start_idx + _fabrics_args_size.obstacle_dim};
        std::string name = "radius_obst_" + std::to_string(i);
        _fabrics_args_idx.push_back(constraint_idx_pair(name, std::pair<uint, uint>(start_idx, end_idx)));
        start_idx = end_idx;
    }
    // weight goal
    for (uint i=0; i<fabrics_config.num_goals_weights; ++i)
    {
        uint end_idx{start_idx + _fabrics_args_size.goal_weight_dim};
        std::string name = "weight_goal_" + std::to_string(i);
        _fabrics_args_idx.push_back(constraint_idx_pair(name, std::pair<uint, uint>(start_idx, end_idx)));
        start_idx = end_idx;
    }
    // goals
    for (uint i=0; i<fabrics_config.num_goals; ++i)
    {
        uint end_idx{start_idx + _fabrics_args_size.goal_dim};
        std::string name = "x_goal_" + std::to_string(i);
        _fabrics_args_idx.push_back(constraint_idx_pair(name, std::pair<uint, uint>(start_idx, end_idx)));
        start_idx = end_idx;
    }
    // obstacle pos
    for (uint i=0; i<fabrics_config.num_obstacles; ++i)
    {
        uint end_idx{start_idx + _fabrics_args_size.obstacle_pos_dim};
        std::string name = "x_obst_" + std::to_string(i);
        _fabrics_args_idx.push_back(constraint_idx_pair(name, std::pair<uint, uint>(start_idx, end_idx)));
        start_idx = end_idx;
    }
}



void
FabricsServer::FillFabricsInputArgs(const std::vector<double>& tcp_recv_data, const double**& fabrics_input)
{
    for (size_t i = 0; i < fabrics_config.fabrics_input_size; ++i) 
    {
        fabrics_input[i] = tcp_recv_data.data() + _fabrics_args_idx[i].second.first;
    }
}


void 
FabricsServer::FillFabricsOutputArgs(std::ostringstream& oss)
{
    for (int i = 0; i < fabrics_config.num_dofs; ++i) {
        oss << _fabrics_output[0][i];
        if (i < fabrics_config.num_dofs - 1) {
            oss << " ";
        }
    }
}


void
FabricsServer::Update()
{
    if (_server->ReadRequest(_tcp_recv_data) ==  TCPClientStatus::NEW_MESSAGE)
    {
        FabricsServer::FillFabricsInputArgs(_tcp_recv_data, _fabrics_input);       
        _fabrics_output[0] = new double[fabrics_config.num_dofs];

        casadi_f0(_fabrics_input, _fabrics_output, _setting_0, _setting_1, _setting_2);
        
        // Sending response to the client
        std::ostringstream oss;
        FabricsServer::FillFabricsOutputArgs(oss);

        if (_server->SendingResponse(oss.str()) == TCPClientStatus::DISCONNECTED)
        {
            std::cout << "Cannot send msg to client. Client disconnected...\n";
            if (_server->WaitForClient() != TCPClientStatus::CONNECTED)
            {
                _server->CloseServer();
            }
        }
    }
    else
    {
        std::cout << "Cannot receive msg from client. Client disconnected.\n";
        if (_server->WaitForClient() != TCPClientStatus::CONNECTED)
        {
            _server->CloseServer();
        }
    }

}