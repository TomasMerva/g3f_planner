#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <sstream> 
#include <memory>
#include <utility>     
#include <csignal>
#include "yaml-cpp/yaml.h"
#include "tcp_server.h"
#include "fabrics_config.h"



class FabricsServer
{
    public:
        FabricsServer(const uint port, 
                      const std::string config_file_path);
        FabricsConfig ReadConfigFile(const std::string config_file_path);
        void PrintFabricsArguments();
        void PrintArgumentsIndices();
        void FillFabricsInputArgs(const std::vector<double>& _tcp_recv_data, const double**& _fabrics_input);
        void FillFabricsOutputArgs(std::ostringstream& oss);
        void Update();
        

        const FabricsConfig fabrics_config;

        
    private:
        std::unique_ptr<TCPServer> _server;

        FabricsArgumentSize _fabrics_args_size;
        typedef std::pair<std::string, std::pair<uint, uint>> constraint_idx_pair;
        std::vector<constraint_idx_pair> _fabrics_args_idx;

        // unknown variables for generated fabrics code
        long long int* _setting_0  = new long long int[2];
        double* _setting_1  = new double[2];
        int _setting_2{0};

        const double** _fabrics_input;
        double** _fabrics_output;

        std::vector<double> _tcp_recv_data;

        void _ComputeParameterIndices();
        static void _CloseConnection(int signal);
};

