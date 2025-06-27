    #include <iostream>
    #include <vector>
    #include <fstream>
    #include <sstream>
    #include <random>
    #include <Eigen/Dense>
    #include <Eigen/QR>
    #include <algorithm>
    #include <iomanip>
    #include <fcntl.h>
    #include <unistd.h>
    #include <termios.h>
    #include <sys/ioctl.h>
    #include <chrono>
    #include <thread>
    #include <cstring>
    #include <sys/time.h>

    using namespace Eigen;
    using namespace std;

    //电池参数
    struct BatteryParam {
        double R0   = 0.072;        // Ω
        double Rp   = 0.03;         // Ω
        double Cp   = 1000;       // F
        double Qn   = 2 * 3600.0; // 容量 (As)
        double deltaT   = 1;          // 采样周期 (s)
        double eta  = 1.0;          // 库仑效率
    };

    /* 状态空间系数矩阵（线性） */
    struct CoeffMatrix {
        Matrix2d A;
        Vector2d B;
        double   D;
    };

    //真实检测数据
    struct Battery {
        double Cur;   // 单个电流数据
        double Vol;   // 单个电压数据
        Battery() = default;
        Battery(double Cur, double Vol) : Cur(Cur), Vol(Vol) {}
    };

    /* —— 噪声参数 —— */
    struct NoiseParam {
        double    ww = 0.0001;   // 过程噪声均匀上界
        double    vv = 0.001;   // 测量噪声均匀上界
        MatrixXd  Gw = Matrix2d::Identity() * 1e-3;   // EKF 用
        double    Gv = 0.01;   // EKF 用
    };

    /* —— 用于求 EKF 增益矩阵—— */
    struct EKFGainParam {
        MatrixXd Q = Matrix2d::Identity() * 1e-8;
        double   R = 0.01;
        MatrixXd P0 = Matrix2d::Zero();  // 初始协方差矩阵
        EKFGainParam() {
            P0(0,0) = 0.1;
            P0(1,1) = 1.0;
        }
    };

    /* —— 状态估计的初始值 —— */
    struct InitParam {
        Vector2d x1 = {0.01, 0.8};  // 初始状态
        Vector2d x1_ = {0.012, 0.7}; // 其他初始值
        MatrixXd BG1_diag = Matrix2d::Zero(); // 边界生成器初值
        InitParam() {
            BG1_diag(0, 0) = 0.03;
            BG1_diag(1, 1) = 0.01;
        }
    };

    /* —— EKF 输出结果打包 —— */
    struct EstimateOutput {
        Vector2d x_hat, x_low, x_up;
        double y_hat, y_low, y_up;
    };

    /* —— 真实系统仿真模型—— */
    struct SimSystem {
        Vector2d x;   // 2×N 真实状态序列
        double y;   // (N-1)×1 真实输出序列
    };   
    
    // 维度结构体
    struct Dimension {
        int s;   // 状态维度
        int n;    // 输出维度
        
        Dimension(int s, int n) : s(s), n(n) {} // 通过构造函数初始化
    };

    //===================================================================================//

    inline CoeffMatrix makeParaMatrix(const BatteryParam& bp)
    {
        CoeffMatrix matrix;                      // 待填充的返回值

        double e = exp(-bp.deltaT / (bp.Rp * bp.Cp));
        matrix.A <<  e , 0, 0 , 1;
        matrix.B <<  bp.Rp * (1 - e), -bp.eta * bp.deltaT / bp.Qn;
        matrix.D  = -bp.R0;

        return matrix;                           // Return-by-value（RVO 会消除拷贝）
    }

    // 串口初始化函数
    int init_serial_port(const char* port, int baudrate = B115200) {
        int fd = open(port, O_RDWR | O_NOCTTY);
        if (fd < 0) {
            return -1;
        }

        struct termios tty;
        if(tcgetattr(fd, &tty) != 0) {
            close(fd);
            return -1;
        }

        cfsetospeed(&tty, baudrate);
        cfsetispeed(&tty, baudrate);

        tty.c_cflag &= ~PARENB; // 无奇偶校验
        tty.c_cflag &= ~CSTOPB; // 1位停止位
        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;     // 8位数据位
        tty.c_cflag &= ~CRTSCTS; // 无硬件流控
        tty.c_cflag |= CREAD | CLOCAL; // 启用接收

        tty.c_lflag &= ~ICANON; // 非规范模式
        tty.c_lflag &= ~ECHO;   // 禁用回显
        tty.c_lflag &= ~ECHOE;
        tty.c_lflag &= ~ISIG;

        tty.c_iflag &= ~(IXON | IXOFF | IXANY); // 无软件流控
        tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL);

        tty.c_oflag &= ~OPOST; // 原始输出
        tty.c_oflag &= ~ONLCR;

        tty.c_cc[VMIN] = 1;    // 读取至少1个字符
        tty.c_cc[VTIME] = 5;   // 0.5秒超时

        if(tcsetattr(fd, TCSANOW, &tty) != 0) {
            close(fd);
            return -1;
        }

        return fd;
    }

    // 解析串口数据 (根据Python代码的协议格式)
    bool parse_serial_data(const string& data, double& voltage, double& current, int& k) {
        // 通道1数据: <ch1>:3998
        // if (data.find("<ch1>:") == 0) {
        //     try {
        //         string value_str = data.substr(6);
        //         int value = stoi(value_str);
        //         voltage = value / 1000.0; // 毫伏转伏特
        //         current = 1.0; // 只有电压值
        //         cout << "开路电压：" << fixed << setprecision(3) << voltage << "V" << endl;
        //         return true;
        //     } catch (...) {
        //         return false;
        //     }
        // }
        // // 全通道数据: <all>:3998,4003,0
        // else 
        if (data.find("<all>:") == 0) {
            try {
                string values_part = data.substr(6);
                vector<string> values;
                stringstream ss(values_part);
                string token;
                
                while (getline(ss, token, ',')) {
                    values.push_back(token);
                }
                
                if (values.size() >= 3) {
                    double U = stoi(values[0]) / 1000.0;  // 电池端电压(V)
                    double U_p = stoi(values[1]) / 1000.0; // 极性电压(V)
                    double Rp = U;
                    double I_k = U_p / Rp;  // 计算电流(A)
                    
                    voltage = U;
                    current = I_k;
                    
                    cout << "U(" << k << "): " << fixed << setprecision(3) << U << "V, "
                        << "I(" << k << "): " << I_k << "A, " ;

                    return true;
                }
            } catch (...) {
                return false;
            }
        }
        return false;
    }

    // 从串口读取一行数据
    string read_serial_line(int fd) {
        const int buf_size = 256;
        char buf[buf_size];
        string line;
        ssize_t n;
        
        while ((n = read(fd, buf, buf_size - 1)) > 0) {
            buf[n] = '\0';
            line += buf;
            
            // 检查是否收到换行符
            if (line.find('\n') != string::npos) {
                // 去掉换行符和可能的回车符
                line.erase(remove(line.begin(), line.end(), '\n'), line.end());
                line.erase(remove(line.begin(), line.end(), '\r'), line.end());
                return line;
            }
        }
        
        return "";
    }


    // 从文件读取向量数据
    VectorXd load_data(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "错误：无法打开文件 " << filename << endl;
            exit(1);
        }
        
        vector<double> data;
        string line;
        while (getline(file, line)) {
            if (!line.empty()) {
                stringstream ss(line);
                double value;
                while (ss >> value) {
                    data.push_back(value);
                }
            }
        }
        file.close();
        
        VectorXd vec(data.size());
        for (int i = 0; i < data.size(); i++) {
            vec(i) = data[i];
        }
        return vec;
    }
    
    /* ---------------------------------------------------------
    *  生成真实状态 & 输出
    *  输入：
    *      - 系数矩阵  coeff         (A, B, D)
    *      - 电流序列  Cur           (N×1)
    *      - 初始状态  init.x1
    *      - OCV 拟合  pfit
    *      - 噪声参数  noise.ww / noise.vv
    *  输出：
    *      - 返回 SimSystem{x, y}
    * --------------------------------------------------------- */
    inline SimSystem generate_state(const CoeffMatrix& matrix, Battery input,
                                    const InitParam& init, const VectorXd& pfit,
                                    const NoiseParam& noise)
    {
        SimSystem sim;
        sim.x = init.x1; 

        random_device rd;
        mt19937       gen(rd());
        uniform_real_distribution<> dis_w(-noise.ww, noise.ww);
        uniform_real_distribution<> dis_v(-noise.vv, noise.vv);

        Vector2d w(dis_w(gen), dis_w(gen));
        sim.x = matrix.A * sim.x + matrix.B * input.Cur + w;
        double soc = sim.x(1);  
        double OCV = 0.0;
        for (int i = 0; i < pfit.size(); ++i)
            OCV += pfit[i] * pow(soc, pfit.size() - 1 - i);
        double v_noise = dis_v(gen);
        sim.y = OCV - sim.x(0) + matrix.D * input.Cur + v_noise;
        // std::cout << 'x=' << sim.x << ' ' << 'y=' << sim.y << std::endl;
        return sim;  
    }


    // 降阶函数
    MatrixXd jiangjie(const MatrixXd& G, int s) {
        int n_z = G.rows(); // 获取行数
        int r = G.cols();   // 获取列数
        
        // 步骤1: 计算各列的欧氏范数并排序
        VectorXd Gn(r);
        for (int j = 0; j < r; ++j) {
            Gn(j) = G.col(j).norm(); // 计算欧氏范数
        }

        // 创建索引并降序排序
        vector<int> indices(r);
        for (int i = 0; i < r; ++i) indices[i] = i;
        sort(indices.begin(), indices.end(), [&](int a, int b) {
            return Gn(a) > Gn(b);
        });

        // 按新顺序排列矩阵
        MatrixXd Gx0(n_z, r);
        for (int j = 0; j < r; ++j) {
            Gx0.col(j) = G.col(indices[j]);
        }

        // 情况1: r <= s 直接返回
        if (r <= s) {
            return Gx0;
        }
        // 情况2: r > s
        else {
            // 错误检查
            if (s < n_z) {
                throw invalid_argument("error! s must be >= n_z");
            }
            
            int k = s - n_z; // 计算要保留的列数
            
            // 分割矩阵为两部分
            MatrixXd Gx01 = Gx0.leftCols(k);            // 前k列
            MatrixXd Gx02 = Gx0.rightCols(r - k);        // 剩余列
            
            // 计算Q矩阵 (对角线为行绝对值和)
            MatrixXd Q = MatrixXd::Zero(n_z, n_z);
            for (int i = 0; i < n_z; ++i) {
                Q(i, i) = Gx02.row(i).cwiseAbs().sum(); // 计算每行绝对值和
            }
            
            // 水平拼接矩阵
            MatrixXd G_j(Gx01.rows(), Gx01.cols() + Q.cols());
            G_j << Gx01, Q;
            
            return G_j;
        }
    }

    // 多项式求值辅助函数
    double polyval(const VectorXd& coeffs, double x) {
        double result = 0.0;
        for (int i = 0; i < coeffs.size(); ++i) {
            result = result * x + coeffs[coeffs.size() - 1 - i];
        }
        return result;
    }

    // 多项式导数系数计算
    VectorXd polyder(const VectorXd& coeffs) {
        if (coeffs.size() <= 1) {
            return VectorXd::Zero(1);
        }
        
        VectorXd deriv(coeffs.size() - 1);
        for (int i = 0; i < deriv.size(); ++i) {
            deriv[i] = coeffs[i] * (coeffs.size() - 1 - i);
        }
        return deriv;
    }

    void save_data_to_csv(const EstimateOutput& est, const SimSystem& sim, int k)
    {
        // 打开文件进行追加
        ofstream file("../data/output/online.csv");
        
        // 文件的第一行标题，仅在文件为空时写入
        static bool is_first_line = true;
        if (is_first_line) {
            file << "k,Up_est,Up_low,Up_up,Up_true,SOC_est,SOC_low,SOC_up,SOC_true,Vol,y_est,y_low,y_up,y_true\n";
            is_first_line = false;
        }

        // 将当前时刻的数据追加到文件
        file << k << ","
            << est.x_hat[0] << ","
            << est.x_low[0] << ","
            << est.x_up[0] << ","
            << sim.x[0] << ","
            << est.x_hat[1] << ","
            << est.x_low[1] << ","
            << est.x_up[1] << ","
            << sim.x[1] << ","
            << sim.y << ","
            << est.y_hat << ","
            << est.y_low << ","
            << est.y_up << ","
            << sim.y << "\n";
    }

    void is_fault(const double& Vol, const double& y_l, const double& y_u, bool fault_status, int k) {     
        fault_status = (Vol < y_l || Vol > y_u) ? 1 : 0;
        // 打印输出格式
        std::cout << "k=" << k << ", f=" << fault_status;// << std::endl;
    }

    // 离线数据集EKF实现
    void EKF_offline(const CoeffMatrix& cm, int s, const VectorXd& pfit, 
                const Battery& input, const NoiseParam& noise, EKFGainParam& gain, 
                EstimateOutput& output, MatrixXd& BG_current)
    { 
        // 预测步骤
        Vector2d x_pre = cm.A * output.x_hat + cm.B * input.Cur;
        MatrixXd BG_pre = cm.A * BG_current;
        BG_pre.conservativeResize(NoChange, BG_pre.cols() + noise.Gw.cols());
        BG_pre.rightCols(noise.Gw.cols()) = noise.Gw;
        
        // 计算测量矩阵
        RowVectorXd C(2);
        C << -1, polyval(polyder(pfit), x_pre[1]);
        
        // 估计输出
        double Uoc = polyval(pfit, x_pre[1]);
        double Up = x_pre[0];
        output.y_hat = Uoc - Up + gain.R * input.Cur;
        
        // 计算输出边界
        MatrixXd BG_y = C * BG_current;
        BG_y.conservativeResize(NoChange, BG_y.cols() + 1);
        BG_y(BG_y.cols() - 1) = noise.Gv;
        BG_y = jiangjie(BG_y, s);
        
        output.y_low = output.y_hat - BG_y.cwiseAbs().sum();
        output.y_up = output.y_hat + BG_y.cwiseAbs().sum();
        
        // 卡尔曼增益和状态更新
        MatrixXd P_pred = cm.A * gain.P0 * cm.A.transpose() + gain.Q;
        double S = (C * P_pred * C.transpose())(0, 0) + gain.R; // 标量计算
        MatrixXd K = P_pred * C.transpose() * (1.0 / S);        // 用除法替代逆
        MatrixXd temp = (MatrixXd::Identity(output.x_hat.size(), output.x_hat.size()) - K * C);
        gain.P0 = temp * P_pred;
        
        // 状态更新
        VectorXd x_updated = x_pre + K * (input.Vol - output.y_hat);
        output.x_hat = x_updated;
        
        // 更新边界生成器
        MatrixXd BG(BG_pre);
        BG.conservativeResize(NoChange, BG.cols() + (K * (-BG_y)).cols());
        BG.rightCols((K * (-BG_y)).cols()) = K * (-BG_y);
        BG_current = jiangjie(BG, s);
        
        // 计算状态边界
        VectorXd abs_sum = BG_current.cwiseAbs().rowwise().sum();
        output.x_low = x_updated - abs_sum;
        output.x_up = x_updated + abs_sum;
        // std::cout << "x_hat=" << output.x_hat << ' x_l=' << output.x_low << ' x_u=' << output.x_up << std::endl;
        // std::cout << "y_hat=" << output.y_hat << ' y_l=' << output.y_low << ' x_u=' << output.y_up << std::endl;
    } 

    int main() {
        BatteryParam bp;   // 电池参数
        CoeffMatrix cm = makeParaMatrix(bp);  //系数矩阵
        VectorXd pfit = load_data("data/pre_data/pfit_online.txt"); // 加载拟合的SOC非线性观测系数C
        Dimension dim(12, 2); // 维数
        InitParam init; // 初始值
        NoiseParam noise; // 噪声
        EKFGainParam gain; //ekf增益矩阵参数

        //加载原始数据
        const char* serial_port = "/dev/ttyUSB0";
        VectorXd voltage, current;
        bool use_serial = false;

        // 尝试打开串口
        int serial_fd = init_serial_port(serial_port);
        if (serial_fd >= 0) {
            cout << "成功打开串口 " << serial_port << " 波特率115200" << endl;
            use_serial = true;
        } else {
            cerr << "未打开串口" << endl;
        }
        
        // 若串口打开，进入主循环
        if (use_serial) {
            cout << "开始串口数据采集，间隔 " << bp.deltaT << " 秒" << endl;
            int k = 1; // 采样时刻计数器
            // 初始化
            double voltage = 0.0, current = 0.0;
            SimSystem sim;
            sim.x = init.x1;
            EstimateOutput output;
            output.x_hat = init.x1_; 
            output.x_low = init.x1_ - init.BG1_diag.cwiseAbs().rowwise().sum();
            output.x_up = init.x1_ + init.BG1_diag.cwiseAbs().rowwise().sum();
            MatrixXd BG_current = init.BG1_diag;

             // 高精度计时变量
            using Clock = std::chrono::steady_clock; // 单调时钟避免系统时间漂移[4,6](@ref)
            auto next_cycle = Clock::now() + std::chrono::seconds(1); // 首次目标时间点

            while (true) {
                // 1. 记录循环开始时刻
                auto cycle_start = Clock::now();

                // 2. 串口数据处理
                string line = read_serial_line(serial_fd);
                // 读取到后，解析
                if (!line.empty()) {
                    // 若成功解析串口数据并赋值
                    if (parse_serial_data(line, voltage, current, k)) {
                        // 原始数据传给输入结构体，噪声处理
                        Battery input(-current, voltage);
                        // 产生soc仿真模型
                        sim = generate_state(cm, input, init, pfit, noise);
                        // 进行EKF状态估计
                        EKF_offline(cm, dim.s, pfit, input, noise, gain, output, BG_current);

                        // 3. 数据保存与输出（严格1秒间隔）
                        save_data_to_csv(output, sim, k);
                        bool fault_status = 0;
                        is_fault(input.Vol, output.y_low, output.y_up, fault_status, k);

                        // 关键输出（确保每次打印间隔1秒）
                        auto now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                        std::cout << std::ctime(&now_time) << "周期" << k 
                                << ": SOC=" << output.x_hat[0] << " 电压=" << voltage << std::endl;

                        k++;
                    }
                }
                // 4. 动态计算并等待剩余时间
                auto cycle_end = Clock::now();
                auto elapsed = cycle_end - cycle_start;
                auto wait_time = std::chrono::duration_cast<std::chrono::nanoseconds>(next_cycle - cycle_end);
                
                // 高精度休眠（纳秒级控制）
                if (wait_time.count() > 0) {
                    struct timespec req = {
                        .tv_sec = static_cast<time_t>(wait_time.count() / 1000000000),
                        .tv_nsec = static_cast<long>(wait_time.count() % 1000000000)
                    };
                    nanosleep(&req, nullptr); // 比sleep_for精度更高[7](@ref)
                } else {
                    std::cerr << "警告：循环超时 " << -wait_time.count() / 1000 << " 微秒" << std::endl;
                }
                
                // 5. 更新下一周期目标时间点
                next_cycle += std::chrono::seconds(1); // 固定步长推进
            }
            close(serial_fd);
        } 
        return 0;
    }

