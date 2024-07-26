#include <iostream>
#include <chrono>
#include "pure_controller.c"

#include <iostream>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <stdlib.h>     /* strtod */
#include <string>
#include <vector>
#include <sstream> 
#include "yaml-cpp/yaml.h"

#define PORT 8080


int main(int argc, char **argv)
{
    YAML::Node config = YAML::LoadFile("../config/fabrics_config.yaml");

    std::cout << config.size() << "\n";
    std::cout << config["problem"].size() << "\n";

    const uint dim_states = 2;
    const uint num_fabrics_goals = config["problem"]["goal"]["goal_definition"].size();
    const uint num_fabrics_goals_weights = config["problem"]["goal"]["goal_definition"].size();
    const uint num_obstacles = config["problem"]["environment"]["number_spheres"]["dynamic"].size() 
        + config["problem"]["environment"]["number_spheres"]["static"].size();
    const uint num_collision_link = config["problem"]["robot_representation"]["collision_links"].size();
    const uint num_planes = config["problem"]["environment"]["number_planes"].size();
    const uint num_dofs = config["problem"]["joint_limits"]["lower_limits"].size();

    const uint fabrics_input_size = num_fabrics_goals +
                                    num_fabrics_goals_weights +
                                    num_obstacles +
                                    num_collision_link +
                                    num_planes +
                                    dim_states;
    const uint fabrics_output_size = 1;

    std::cout << "input size: " << fabrics_input_size <<"\n";
    std::cout << "output size: " << fabrics_output_size <<"\n";


    // Fabrics Raw message
    uint current_idx = 0;
    std::vector<uint> idx_fabrics_args{0};
    // Plane constraint
    for (uint i=0; i<num_planes; ++i)
    {
        current_idx += 4;
        idx_fabrics_args.push_back(current_idx);
    }
    // q and qdot
    for (uint i=0; i<dim_states; ++i)
    {
        current_idx += num_dofs;
        idx_fabrics_args.push_back(current_idx);
    }
    // radius body
    for (uint i=0; i<num_collision_link; ++i)
    {
        current_idx += 1;
        idx_fabrics_args.push_back(current_idx);
    }
    // radius obstacle
    for (uint i=0; i<num_obstacles; ++i)
    {
        current_idx += 1;
        idx_fabrics_args.push_back(current_idx);
    }
    // weight goal


    // subgoals

    // obstacle position


    //=========================

    // long long int* setting_0 = new long long int[2];
    // double* setting_1 = new double[2];
    // int setting_2 = 0;

    // const double** fabrics_input = new const double*[4];
    // double** fabrics_output = new double*[1];

    // int server_fd, new_socket;
    // struct sockaddr_in address;
    // int addrlen = sizeof(address);
    // char buffer[1024] = {0};
    // const char* hello = "Hello from server";

    // // Creating socket file descriptor
    // if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
    //     perror("socket failed");
    //     exit(EXIT_FAILURE);
    // }

    // // Setting up the address structure
    // address.sin_family = AF_INET;
    // address.sin_addr.s_addr = INADDR_ANY;
    // address.sin_port = htons(PORT);

    // // Binding the socket to the port
    // if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
    //     perror("bind failed");
    //     close(server_fd);
    //     exit(EXIT_FAILURE);
    // }
    // std::cout << "Waiting for client\n";

    // // Listening for connections
    // if (listen(server_fd, 3) < 0) {
    //     perror("listen");
    //     close(server_fd);
    //     exit(EXIT_FAILURE);
    // }
    // std::cout << "Client connected.\n";

    // // Accepting a connection
    // if ((new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {
    //     perror("accept");
    //     close(server_fd);
    //     exit(EXIT_FAILURE);
    // }

    // while (1)
    // {
    //     // Reading from the client
    //     memset(&buffer, 0, sizeof(buffer));//clear the buffer
    //     int bytes_read = read(new_socket, buffer, 1024);
    //     if (bytes_read>0)
    //     {
    //         std::stringstream ss(buffer);
    //         std::vector<double> fabrics_raw_data;
    //         std::string word;
    //         while (ss >> word)
    //         {
    //             fabrics_raw_data.push_back(std::stod(word));
    //         }


        
    //         // Fabrics -----------------
    //         int n_dof = 9;

    //         // q,dq,w,eef
    //         const int q_IDX = 0;
    //         const int dq_IDX = n_dof;
    //         const int weight_IDX = 2*n_dof;
    //         const int eef_IDX = 2*n_dof+1;
    //         std::vector<double> q = std::vector<double>(fabrics_raw_data.begin() +q_IDX, fabrics_raw_data.begin() + dq_IDX);
    //         std::vector<double> dq = std::vector<double>(fabrics_raw_data.begin()+ dq_IDX, fabrics_raw_data.begin()+weight_IDX );
    //         std::vector<double> weight = std::vector<double>(fabrics_raw_data.begin()+ weight_IDX, fabrics_raw_data.begin() + weight_IDX +1);
    //         std::vector<double> eef = std::vector<double>(fabrics_raw_data.begin() + eef_IDX, fabrics_raw_data.end());


    //         fabrics_input[0] = q.data();
    //         fabrics_input[1] = dq.data();
    //         fabrics_input[2] = weight.data();
    //         fabrics_input[3] = eef.data();
    //         fabrics_output[0] = new double[n_dof];

    //         casadi_f0(fabrics_input, fabrics_output, setting_0, setting_1, setting_2);
        
    //         // Sending response to the client
    //         std::ostringstream oss;
    //         for (int i = 0; i < n_dof; ++i) {
    //             oss << fabrics_output[0][i];
    //             if (i < n_dof - 1) {
    //                 oss << " ";
    //             }
    //         }
    //         std::strcpy(buffer, oss.str().c_str()); 
    //         send(new_socket, buffer, strlen(buffer), 0);
    //     }
    // }
    

    // // Closing the socket
    // close(new_socket);
    // close(server_fd);


   
    return 0;
}